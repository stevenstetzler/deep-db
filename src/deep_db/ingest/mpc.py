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

def _upsert_mpc_rows(session, rows):
    """
    Given row dicts with
    obsid/exposure_id/detector_exposure_id/designation/trkid/trksub/ra/dec/obstime/mag/status
    (as parsed from a query_mpc PSV stream):

    1. Resolve each row's designation to an existing MPCDesignation, or
       bulk-insert a new one (see _resolve_designations).
    2. Fetch the MPCTracklet rows already in the database for every
       (trkid, trksub) pair the rows cover (in batches, same reasoning as
       _upsert_mpchecker_rows), then bulk-insert a new MPCTracklet for each
       pair not already present, or update the existing one's
       designation_id if it's changed (MPC's identification of a
       tracklet's object can be revised over time).
    3. Fetch the MPCObservation rows already in the database for every
       obsid the rows cover (also batched), then bulk-insert a new
       MPCObservation for each obsid not already present, or update the
       existing one's designation_id/detector_exposure_id if either is
       changed. obsid is an immutable PK, so ra/dec/obstime/mag/status (and
       exposure_id, a pure function of obstime) are taken as fixed once
       inserted -- only the designation attribution and detector_exposure_id
       (should a re-ingest's bounds check match a different DetectorExposure,
       or go from/to a miss) are ever revised.

    Tracklet fitting (deriving a position/velocity Tracklet per
    MPCTracklet, and merging coincident ones into a MergedMPCTracklet) is
    left for later, as with _upsert_mpchecker_rows.

    Returns the {designation tuple: id} lookup used to build them.
    """
    from sqlalchemy import tuple_
    from deep_db.database.mpc import MPCTracklet, MPCObservation
    from deep_db.ingest.designations import _resolve_designations

    if not rows:
        return {}

    object_ids = _resolve_designations(session, {r['designation'] for r in rows})

    tracklet_keys = {(r['trkid'], r['trksub']) for r in rows}
    existing_tracklets = {}
    for batch in _chunked(tracklet_keys, 500):
        for t in session.query(MPCTracklet).filter(
            tuple_(MPCTracklet.trkid, MPCTracklet.trksub).in_(batch)
        ):
            existing_tracklets[(t.trkid, t.trksub)] = t

    new_tracklets = []
    for r in rows:
        key = (r['trkid'], r['trksub'])
        object_id = object_ids[r['designation']]
        existing = existing_tracklets.get(key)
        if existing is None:
            tracklet = MPCTracklet(object_id=object_id, trkid=key[0], trksub=key[1])
            new_tracklets.append(tracklet)
            existing_tracklets[key] = tracklet  # dedup repeated keys within this batch
        elif existing.object_id != object_id:
            existing.object_id = object_id
    if new_tracklets:
        session.add_all(new_tracklets)
        session.flush()  # populate ids for newly created tracklets
    session.commit()

    tracklet_ids = {key: t.id for key, t in existing_tracklets.items()}

    obsids = {r['obsid'] for r in rows}
    existing_observations = {}
    for batch in _chunked(obsids, 500):
        for o in session.query(MPCObservation).filter(MPCObservation.obsid.in_(batch)):
            existing_observations[o.obsid] = o

    new_observations = []
    for r in rows:
        object_id = object_ids[r['designation']]
        existing = existing_observations.get(r['obsid'])
        if existing is None:
            observation = MPCObservation(
                obsid=r['obsid'],
                exposure_id=r['exposure_id'],
                detector_exposure_id=r['detector_exposure_id'],
                mpc_tracklet_id=tracklet_ids[(r['trkid'], r['trksub'])],
                object_id=object_id,
                ra=r['ra'],
                dec=r['dec'],
                mag=r['mag'],
                obstime=r['obstime'],
                status=r['status'],
            )
            new_observations.append(observation)
            existing_observations[r['obsid']] = observation  # dedup repeated obsids within this batch
        else:
            if existing.object_id != object_id:
                existing.object_id = object_id
            if existing.detector_exposure_id != r['detector_exposure_id']:
                existing.detector_exposure_id = r['detector_exposure_id']
    if new_observations:
        session.add_all(new_observations)
    session.commit()

    return object_ids

def ingest_mpc(session, table):
    from sqlalchemy import tuple_
    from astropy.time import Time
    from deep_db.database.survey import Exposure, Detector, DetectorExposure
    expnums = [_type_or_none(v, int) for v in table['expnum']]
    detnums = [_type_or_none(v, int) for v in table['detnum']]
    table['mag'] = [_type_or_none(v, float) for v in table['mag']]

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
                obsid=row['obsid'],
                exposure_id=expid_lookup.get(expnum),
                detector_exposure_id=detexpid_lookup.get((expnum, detnum)),
                designation=(
                    row['provid'] or None,
                    row['permid'] or None,
                    row['name'] or None,
                ),
                trksub=row['trksub'] or None,
                trkid=row['trkid'] or None,
                ra=row['ra'],
                dec=row['dec'],
                obstime=Time(row['obstime'], scale='utc').utc.datetime,
                mag=row['mag'],
                status=row['status'],
            )
        )
    _upsert_mpc_rows(session, rows)
