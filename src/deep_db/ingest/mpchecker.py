def _type_or_none(value, _type):
    import numpy as np
    """Plain int, or None for None / masked / NaN."""
    if value is None or value is np.ma.masked or (isinstance(value, str) and (value.strip() == '' or value.strip() == 'None')):
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return _type(value)

def _chunked(items, size):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]

def _upsert_mpchecker_rows(session, rows):
    """
    Given row dicts with
    designation/exposure_id/detector_exposure_id/ra/dec/time/v_mag (as
    produced by `_mpchecker_rows_for_exposure` or parsed from a
    query_mpchecker PSV stream):

    1. Query every existing MPCDesignation.
    2. Resolve each row's designation to one of them, or bulk-insert a new
       MPCDesignation if it doesn't match any (see _resolve_designations).
    3. Fetch the MPCheckerEphemeris rows already in the database for every
       (designation, exposure_id, detector_exposure_id) triple the rows
       cover -- in batches, since a single `(a, b, c) IN ((..), (..), ...)`
       filter over every triple from a whole-survey ingest is large enough
       to blow Postgres's parser stack -- then bulk-insert a new
       MPCheckerEphemeris for each triple not already present, or update
       the existing one's ra/dec/v_mag. detector_exposure_id is nullable
       (an object that missed every DetectorExposure's bounds), and SQL's
       row-value IN/`=` never matches a NULL against another NULL, so rows
       with detector_exposure_id IS NULL are looked up in a separate pass,
       keyed on (designation, exposure_id) alone with an explicit IS NULL
       filter -- a plain tuple_(...).in_(batch) over triples that include
       None would silently never find them, inserting a duplicate "missed"
       row on every re-ingest.

    A row's `time` is dropped here -- MPCheckerEphemeris.time is a
    read-only property derived from `exposure`.midpoint, not a column,
    since it's always just that Exposure's midpoint (exposure_id, unlike
    detector_exposure_id, is always set, so this works for a miss too).

    Returns the {designation tuple: id} lookup used to build them (covering
    every row's designation, whether newly inserted or matched to an
    existing one).
    """
    from sqlalchemy import tuple_
    from deep_db.ingest.designations import _resolve_designations
    from deep_db.database.mpchecker import MPCheckerEphemeris

    if not rows:
        return {}

    object_ids = _resolve_designations(
        session, {r['designation'] for r in rows}
    )

    ephemeris_keys = {
        (object_ids[r['designation']], r['exposure_id'], r['detector_exposure_id'])
        for r in rows
    }
    matched_keys = {k for k in ephemeris_keys if k[2] is not None}
    missed_keys = {(k[0], k[1]) for k in ephemeris_keys if k[2] is None}

    existing_ephemeris = {}
    for batch in _chunked(matched_keys, 500):
        for e in session.query(MPCheckerEphemeris).filter(
            tuple_(
                MPCheckerEphemeris.object_id,
                MPCheckerEphemeris.exposure_id,
                MPCheckerEphemeris.detector_exposure_id,
            ).in_(batch)
        ):
            existing_ephemeris[(e.object_id, e.exposure_id, e.detector_exposure_id)] = e
    for batch in _chunked(missed_keys, 500):
        for e in session.query(MPCheckerEphemeris).filter(
            tuple_(
                MPCheckerEphemeris.object_id,
                MPCheckerEphemeris.exposure_id,
            ).in_(batch),
            MPCheckerEphemeris.detector_exposure_id.is_(None),
        ):
            existing_ephemeris[(e.object_id, e.exposure_id, None)] = e

    new_ephemeris = []
    for r in rows:
        key = (object_ids[r['designation']], r['exposure_id'], r['detector_exposure_id'])
        existing = existing_ephemeris.get(key)
        if existing is not None:
            existing.ra = r['ra']
            existing.dec = r['dec']
            existing.v_mag = r['v_mag']
            continue
        ephemeris = MPCheckerEphemeris(
            object_id=key[0],
            exposure_id=key[1],
            detector_exposure_id=key[2],
            ra=r['ra'],
            dec=r['dec'],
            v_mag=r['v_mag'],
        )
        new_ephemeris.append(ephemeris)
        existing_ephemeris[key] = ephemeris  # dedup repeated keys within this batch
    if new_ephemeris:
        session.add_all(new_ephemeris)
    session.commit()

    return object_ids

def ingest_mpchecker(session, table):
    from sqlalchemy import tuple_
    from astropy.time import Time
    from deep_db.database.survey import Exposure, Detector, DetectorExposure
    expnums = [_type_or_none(v, int) for v in table['expnum']]
    detnums = [_type_or_none(v, int) for v in table['detnum']]
    table['v_mag'] = [_type_or_none(v, float) for v in table['v_mag']]

    # expnum -> Exposure.id
    unique_expnums = {e for e in expnums if e is not None}
    expid_lookup = {}
    for batch in _chunked(unique_expnums, 500):
        q = session.query(Exposure.expnum, Exposure.id).filter(Exposure.expnum.in_(batch))
        for expnum, exposure_id in q:
            expid_lookup[expnum] = exposure_id

    # (expnum, detnum) -> DetectorExposure.id
    unique_pairs = {(e, d) for e, d in zip(expnums, detnums) if e is not None and d is not None}
    detexpid_lookup = {}
    for batch in _chunked(unique_pairs, 500):
        q = (
            session.query(Exposure.expnum, Detector.number, DetectorExposure.id)
            .select_from(DetectorExposure)
            .join(Exposure, Exposure.id == DetectorExposure.exposure_id)
            .join(Detector, Detector.id == DetectorExposure.detector_id)
            .filter(tuple_(Exposure.expnum, Detector.number).in_(batch))
        )
        for expnum, detnum, detexp_id in q:
            detexpid_lookup[(expnum, detnum)] = detexp_id

    rows = []
    for row, expnum, detnum in zip(table, expnums, detnums):
        rows.append(
            dict(
                exposure_id=expid_lookup.get(expnum),
                detector_exposure_id=detexpid_lookup.get((expnum, detnum)),
                designation=(
                    row['provid'] or None,
                    row['permid'] or None,
                    row['name'] or None,
                ),
                ra=row['ra'],
                dec=row['dec'],
                v_mag=row['v_mag'],
            )
        )

    _upsert_mpchecker_rows(session, rows)
