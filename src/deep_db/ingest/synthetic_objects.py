import logging
logger = logging.getLogger(__name__)

_FAKES_POPULATIONS = {
    # source -> (orbits_file, detections_file), both relative to the
    # data_dir passed to ingest_fakes (data/fakes/DEEP in this repo)
    'asteroid': ('asteroidorbits.fits', 'fakeasteroids.fits'),
    'tno': ('fakes_orbits.fits', 'fakes_detections.fits'),
}
_FAKES_BINARY_PROPERTIES_FILE = 'fakes_binaryproperties.fits'
_FAKES_BINARY_PROPERTIES_SOURCE = 'tno'  # every ORBITID in that file falls within the 'tno' population
_FAKES_ASTEROID_DESIGNATION_OFFSET = 10_000_000  # see _fakes_designation_orbit_id

def _fakes_designation_orbit_id(source, orbit_id):
    """
    Map a (source, orbit_id) pair into the single space
    SyntheticObjectDesignation.orbit_id lives in. The two populations' raw
    ORBITID ranges overlap (both run from roughly 0 to 6.4M), so this
    follows the DEEP fakes pipeline's own disambiguation convention (seen
    in fakes_detections_joined.fits): 'tno' ORBITID is kept as-is,
    'asteroid' ORBITID is offset by 10,000,000.
    """
    if source == 'asteroid':
        return orbit_id + _FAKES_ASTEROID_DESIGNATION_OFFSET
    return orbit_id

def _ingest_fakes_orbits(session, source, path):
    """
    Read every row of an orbits FITS table (asteroidorbits.fits or
    fakes_orbits.fits -- one row per ORBITID, giving Keplerian elements, a
    state vector, and photometric/size parameters for a fake object) and
    bulk-insert a new Elements + State + SyntheticObjectDesignation +
    'primary'-component SyntheticObject for every ORBITID not already
    ingested for this `source`. (Binaries' 'companion' rows are created
    separately, by _ingest_fakes_binary_properties, once this has run.)

    Returns the {orbit_id: synthetic_object_id} lookup, covering every
    ORBITID in the file and pointing at its 'primary' row, whether newly
    inserted or already present.
    """
    from astropy.io import fits
    from deep_db.database.models import Element, State
    from deep_db.database.synthetic_objects import SyntheticObject, SyntheticObjectDesignation

    existing_orbit_ids = {
        orbit_id for (orbit_id,) in
        session.query(SyntheticObject.orbit_id)
        .filter(SyntheticObject.source == source, SyntheticObject.component == "primary")
    }
    designations_by_orbit_id = {
        designation_orbit_id: designation_id
        for designation_orbit_id, designation_id in
        session.query(SyntheticObjectDesignation.orbit_id, SyntheticObjectDesignation.id)
    }

    new_objects = []
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        for row in data:
            orbit_id = int(row['ORBITID'])
            if orbit_id in existing_orbit_ids:
                continue
            existing_orbit_ids.add(orbit_id)
            xv = row['xv']

            designation_orbit_id = _fakes_designation_orbit_id(source, orbit_id)
            designation_id = designations_by_orbit_id.get(designation_orbit_id)
            designation_kwargs = (
                dict(designation_id=designation_id) if designation_id is not None
                else dict(designation=SyntheticObjectDesignation(orbit_id=designation_orbit_id))
            )

            new_objects.append(SyntheticObject(
                source=source,
                orbit_id=orbit_id,
                component="primary",
                element=Element(
                    a=float(row['a']), e=float(row['e']), i=float(row['i']),
                    node=float(row['Omega']), argperi=float(row['omega']), peri_time=float(row['T_p']),
                ),
                state=State(
                    x=float(xv[0]), y=float(xv[1]), z=float(xv[2]),
                    vx=float(xv[3]), vy=float(xv[4]), vz=float(xv[5]),
                ),
                n_images=int(row['N_IMAGES']),
                m_vr=float(row['m_VR']),
                h_vr=float(row['H_VR']),
                d=float(row['d']),
                r=float(row['r']),
                amplitude=float(row['AMP']),
                phase=float(row['PHASE']),
                period=float(row['PERIOD']),
                **designation_kwargs,
            ))

    if new_objects:
        session.add_all(new_objects)
        session.commit()

    return {
        orbit_id: synthetic_object_id
        for orbit_id, synthetic_object_id in session.query(
            SyntheticObject.orbit_id, SyntheticObject.id
        ).filter(SyntheticObject.source == source, SyntheticObject.component == "primary")
    }

def _ingest_fakes_binary_properties(session, path, source):
    """
    Read fakes_binaryproperties.fits (one row per ORBITID that was
    simulated as a binary, giving the companion's DELTA_H/SEPARATION/ANGLE)
    and, for every ORBITID not already linked, create its 'companion'
    SyntheticObject plus the SyntheticObjectBinary linking it to the
    'primary' -- matched against `source`'s already-ingested primary rows.

    The companion shares the primary's element_id/state_id/designation_id
    (ORBITID tracks the binary's barycenter, not either individual body,
    so there's only one orbit/state to point at) and its d/r/n_images/
    amplitude/phase/period (also barycenter-level quantities); only its
    magnitude (m_vr/h_vr) differs, offset from the primary's by DELTA_H.
    """
    from astropy.io import fits
    from deep_db.database.synthetic_objects import SyntheticObject, SyntheticObjectBinary

    primaries = {
        orbit_id: obj
        for (orbit_id, obj) in (
            (obj.orbit_id, obj) for obj in
            session.query(SyntheticObject).filter(
                SyntheticObject.source == source, SyntheticObject.component == "primary",
            )
        )
    }
    existing = {
        primary_id for (primary_id,) in
        session.query(SyntheticObjectBinary.primary_synthetic_object_id)
    }

    new_companions = []
    pending_links = []  # (companion, primary_id, delta_h, separation, angle), linked after flush assigns companion.id
    skipped = 0
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        for row in data:
            orbit_id = int(row['ORBITID'])
            primary = primaries.get(orbit_id)
            if primary is None:
                skipped += 1
                continue
            if primary.id in existing:
                continue
            existing.add(primary.id)
            delta_h = float(row['DELTA_H'])
            companion = SyntheticObject(
                source=source,
                orbit_id=orbit_id,
                component="companion",
                designation_id=primary.designation_id,
                element_id=primary.element_id,
                state_id=primary.state_id,
                n_images=primary.n_images,
                m_vr=primary.m_vr + delta_h,
                h_vr=primary.h_vr + delta_h,
                d=primary.d,
                r=primary.r,
                amplitude=primary.amplitude,
                phase=primary.phase,
                period=primary.period,
            )
            new_companions.append(companion)
            pending_links.append((companion, primary.id, delta_h, float(row['SEPARATION']), float(row['ANGLE'])))

    if new_companions:
        session.add_all(new_companions)
        session.flush()  # assigns companion.id, needed below
        for companion, primary_id, delta_h, separation, angle in pending_links:
            session.add(SyntheticObjectBinary(
                primary_synthetic_object_id=primary_id,
                companion_synthetic_object_id=companion.id,
                delta_h=delta_h,
                separation=separation,
                angle=angle,
            ))
        session.commit()
    if skipped:
        logger.warning(
            "fakes_binaryproperties: skipped %d rows whose ORBITID has no "
            "primary SyntheticObject with source=%r", skipped, source,
        )

def _ingest_fakes_detections(session, source, path, orbit_ids, chunk_size=200_000):
    """
    Stream a detections FITS table (fakeasteroids.fits or
    fakes_detections.fits -- one row per detection of a fake object on a
    single exposure/CCD) in chunks of `chunk_size` rows -- reading the
    whole table at once would mean materializing millions of rows -- and
    bulk-insert a new SyntheticObjectEphemeris for every
    (synthetic_object_id, detector_exposure_id) pair not already present.

    For a binary's ORBITID, the file carries two rows per (EXPNUM, CCDNUM)
    -- the primary and companion detected at the barycenter's position,
    offset -- both tagged with the same (primary's) ORBITID, since there's
    no second ORBITID for the companion. These are disambiguated by file
    order: the primary's row always comes first and is always the
    brighter of the two by exactly its SyntheticObjectBinary.delta_h (true
    for every one of the ~220k such pairs in fakes_detections.fits, so this
    is a safe rule, not a heuristic) -- the first row seen for a given
    (ORBITID, EXPNUM, CCDNUM) is routed to the primary's synthetic_object_id,
    the second to the companion's, with the MAG-difference-vs-delta_h check
    below as a cheap guard against that assumption ever breaking.

    Each row's ORBITID is resolved against `orbit_ids` (the {orbit_id:
    synthetic_object_id} lookup _ingest_fakes_orbits returns for this same
    `source`, pointing at the 'primary' row); its (EXPNUM, CCDNUM) is
    resolved to an already-ingested DetectorExposure the same way
    `ingest_survey` built it (Exposure.expnum, Detector.number), and its
    EXPNUM alone to that DetectorExposure's Exposure (SyntheticObjectEphemeris.
    exposure_id -- every detection belongs to the Exposure table of obstimes
    it was evaluated against, same as detector_exposure_id implies). A row
    whose ORBITID or (EXPNUM, CCDNUM) doesn't resolve is skipped (and
    counted) rather than raising -- the latter happens when `deep-db
    ingest survey` hasn't been run yet for that exposure; re-running this
    ingest after a survey backfill will pick up anything skipped for that
    reason.

    aei/mjd_mid/TDB aren't stored: aei duplicates the orbit's own elements
    row-for-row, and TDB is the integrator's internal time variable
    (decimal years since J2000 in the TDB scale, not a date) rather than
    the row's actual observation epoch -- SyntheticObjectEphemeris.time is
    read through `exposure` instead, the same convention MPCheckerEphemeris
    uses.
    """
    from astropy.io import fits
    from deep_db.database.survey import Detector, Exposure, DetectorExposure
    from deep_db.database.synthetic_objects import SyntheticObject, SyntheticObjectBinary, SyntheticObjectEphemeris

    exposure_ids_by_expnum = dict(session.query(Exposure.expnum, Exposure.id))

    detector_exposure_ids = {
        (expnum, number): detector_exposure_id
        for expnum, number, detector_exposure_id in session.query(
            Exposure.expnum, Detector.number, DetectorExposure.id
        ).join(DetectorExposure, DetectorExposure.exposure_id == Exposure.id)
         .join(Detector, Detector.id == DetectorExposure.detector_id)
    }

    existing_pairs = {
        (synthetic_object_id, detector_exposure_id)
        for synthetic_object_id, detector_exposure_id in session.query(
            SyntheticObjectEphemeris.synthetic_object_id,
            SyntheticObjectEphemeris.detector_exposure_id,
        )
    }

    # Binary orbit_ids need their companion's synthetic_object_id and
    # delta_h on hand to disambiguate/sanity-check the detections file's
    # duplicated (ORBITID, EXPNUM, CCDNUM) rows below; empty for 'asteroid'
    # (no binaries in that population).
    companion_ids_by_orbit_id = {}
    delta_h_by_orbit_id = {}
    for primary_orbit_id, companion_id, delta_h in (
        session.query(
            SyntheticObject.orbit_id,
            SyntheticObjectBinary.companion_synthetic_object_id,
            SyntheticObjectBinary.delta_h,
        )
        .join(SyntheticObjectBinary, SyntheticObjectBinary.primary_synthetic_object_id == SyntheticObject.id)
        .filter(SyntheticObject.source == source, SyntheticObject.component == "primary")
    ):
        companion_ids_by_orbit_id[primary_orbit_id] = companion_id
        delta_h_by_orbit_id[primary_orbit_id] = delta_h

    # (orbit_id, expnum, ccdnum) -> primary row's MAG, awaiting the
    # companion row that completes the pair; persists across chunks since
    # the two rows aren't guaranteed to land in the same chunk.
    pending_primary_mag = {}
    skipped_no_orbit = 0
    skipped_no_detector_exposure = 0
    mismatched_delta_h = 0

    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        n = len(data)
        for start in range(0, n, chunk_size):
            chunk = data[start:start + chunk_size]
            new_rows = []
            for row in chunk:
                orbit_id = int(row['ORBITID'])
                mag = float(row['MAG'])
                companion_id = companion_ids_by_orbit_id.get(orbit_id)
                if companion_id is None:
                    synthetic_object_id = orbit_ids.get(orbit_id)
                else:
                    key = (orbit_id, int(row['EXPNUM']), int(row['CCDNUM']))
                    primary_mag = pending_primary_mag.pop(key, None)
                    if primary_mag is None:
                        pending_primary_mag[key] = mag
                        synthetic_object_id = orbit_ids.get(orbit_id)
                    else:
                        synthetic_object_id = companion_id
                        if abs((mag - primary_mag) - delta_h_by_orbit_id[orbit_id]) > 0.01:
                            mismatched_delta_h += 1
                if synthetic_object_id is None:
                    skipped_no_orbit += 1
                    continue
                detector_exposure_id = detector_exposure_ids.get(
                    (int(row['EXPNUM']), int(row['CCDNUM']))
                )
                if detector_exposure_id is None:
                    skipped_no_detector_exposure += 1
                    continue
                exposure_id = exposure_ids_by_expnum[int(row['EXPNUM'])]
                pair = (synthetic_object_id, detector_exposure_id)
                if pair in existing_pairs:
                    continue
                existing_pairs.add(pair)
                xv = row['xv']
                new_rows.append(dict(
                    synthetic_object_id=synthetic_object_id,
                    exposure_id=int(exposure_id),
                    detector_exposure_id=int(detector_exposure_id),
                    ra=float(row['RA']),
                    dec=float(row['DEC']),
                    mag=mag,
                    d=float(row['d']),
                    r=float(row['r']),
                    x=float(xv[0]), y=float(xv[1]), z=float(xv[2]),
                    vx=float(xv[3]), vy=float(xv[4]), vz=float(xv[5]),
                ))
            if new_rows:
                session.bulk_insert_mappings(SyntheticObjectEphemeris, new_rows)
                session.commit()
            logger.info(
                "%s detections: %d/%d rows processed (%d skipped: no orbit, %d skipped: no detector_exposure)",
                source, min(start + chunk_size, n), n, skipped_no_orbit, skipped_no_detector_exposure,
            )

    if skipped_no_detector_exposure:
        logger.warning(
            "%s: %d detections skipped -- no ingested DetectorExposure matches their "
            "(EXPNUM, CCDNUM); run `deep-db ingest survey` for the missing exposures "
            "and re-run `deep-db ingest fakes` to pick them up",
            source, skipped_no_detector_exposure,
        )
    if skipped_no_orbit:
        logger.warning(
            "%s: %d detections skipped -- no SyntheticObject matches their ORBITID",
            source, skipped_no_orbit,
        )
    if mismatched_delta_h:
        logger.warning(
            "%s: %d companion detections had a MAG - primary_MAG that didn't match "
            "their SyntheticObjectBinary.delta_h within 0.01 mag -- the primary/"
            "companion row-order assumption may not hold for these",
            source, mismatched_delta_h,
        )
    if pending_primary_mag:
        logger.warning(
            "%s: %d binary primary detections never found their companion's row "
            "in this file (orphaned (orbit_id, expnum, ccdnum) keys)",
            source, len(pending_primary_mag),
        )

def ingest_fakes(session, data_dir):
    """
    Ingest the DEEP fakes-injection pipeline's FITS tables from `data_dir`
    (the layout of data/fakes/DEEP in this repo: asteroidorbits.fits,
    fakeasteroids.fits, fakes_orbits.fits, fakes_detections.fits, and
    optionally fakes_binaryproperties.fits) into synthetic_object_designation,
    synthetic_object (+ its element/state), synthetic_object_binary, and
    synthetic_object_ephemeris.

    There are two independent fake-object populations, each a matched
    (orbits, detections) file pair -- 'asteroid' (asteroidorbits.fits +
    fakeasteroids.fits) and 'tno' (fakes_orbits.fits + fakes_detections.fits)
    -- tagged via SyntheticObject.source, since ORBITID is only unique
    *within* one population (the two populations' ORBITID ranges overlap).
    fakes_detections_joined.fits is a derived merge of the two detections
    files (it offsets the asteroid population's ORBITID by 10,000,000 to
    disambiguate) and isn't read here; the two raw per-population pairs are
    ingested directly instead, each orbits file before its matching
    detections file so every detection's ORBITID already resolves. Every
    SyntheticObject gets a SyntheticObjectDesignation using that same
    10,000,000 offset, unifying ORBITID across both populations into one
    identity space (see _fakes_designation_orbit_id).

    fakes_binaryproperties.fits, if present, is ingested after both
    populations' orbits and before either detections file (every ORBITID
    in it falls within the 'tno' population): it adds a 'companion'
    SyntheticObject (sharing the 'primary' row's element/state/designation,
    since ORBITID tracks the binary's barycenter rather than either body)
    plus the SyntheticObjectBinary linking the two, so that when the
    detections file is ingested afterwards, both of a binary's rows per
    exposure already resolve to distinct synthetic_object_ids.
    """
    import os

    orbit_ids_by_source = {}
    for source, (orbits_file, _) in _FAKES_POPULATIONS.items():
        orbits_path = os.path.join(data_dir, orbits_file)
        logger.info("ingesting %s orbits from %s", source, orbits_path)
        orbit_ids_by_source[source] = _ingest_fakes_orbits(session, source, orbits_path)

    binary_properties_path = os.path.join(data_dir, _FAKES_BINARY_PROPERTIES_FILE)
    if os.path.exists(binary_properties_path):
        logger.info("ingesting binary properties from %s", binary_properties_path)
        _ingest_fakes_binary_properties(
            session, binary_properties_path, source=_FAKES_BINARY_PROPERTIES_SOURCE
        )

    for source, (_, detections_file) in _FAKES_POPULATIONS.items():
        detections_path = os.path.join(data_dir, detections_file)
        logger.info("ingesting %s detections from %s", source, detections_path)
        _ingest_fakes_detections(session, source, detections_path, orbit_ids_by_source[source])


