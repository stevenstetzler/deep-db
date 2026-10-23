from deep_db.query.common import autojoin, convert_fields

def query_synthetic_objects(session, fields, orbit_id=[], name=[], source=[], component=[], night=[], field=[], expnum=[], detnum=[], detname=[]):
    from deep_db.database.synthetic_objects import SyntheticObjectDesignation, SyntheticObject, SyntheticObjectEphemeris
    from deep_db.database.survey import Night, Field, Exposure, Detector, DetectorExposure
    tables = [SyntheticObjectDesignation, SyntheticObject, SyntheticObjectEphemeris, Exposure, Night, Field, Detector, DetectorExposure]
    # (table, condition) for each active filter
    filters = []
    if orbit_id:
        filters.append((SyntheticObjectDesignation, SyntheticObjectDesignation.orbit_id.in_(orbit_id)))
    if name:
        filters.append((SyntheticObjectDesignation, SyntheticObjectDesignation.name.in_(name)))
    if source:
        filters.append((SyntheticObject, SyntheticObject.source.in_(source)))
    if component:
        filters.append((SyntheticObject, SyntheticObject.component.in_(component)))
    if night:
        filters.append((Night, Night.night.in_(night)))
    if field:
        filters.append((Field, Field.name.in_(field)))
    if expnum:
        filters.append((Exposure, Exposure.expnum.in_(expnum)))
    if detnum:
        filters.append((Detector, Detector.number.in_(detnum)))
    if detname:
        filters.append((Detector, Detector.name.in_(detname)))

    columns = convert_fields(fields, tables)

    # see query_survey's identical exclusion -- ties SyntheticObjectEphemeris
    # to the specific DetectorExposure it was predicted on instead of every
    # DetectorExposure its Exposure happens to have, whenever Detector/
    # DetectorExposure is involved at all (filtered on, or just selected).
    exclude_edges = []
    if detnum or detname or any(c.class_ in (Detector, DetectorExposure) for c in columns):
        exclude_edges.append((SyntheticObjectEphemeris, Exposure))

    required = [c.class_ for c in columns] + [t for t, _ in filters]
    q = autojoin(session.query(*columns), required, tables, exclude=exclude_edges)
    for _, cond in filters:
        q = q.filter(cond)
    return q.distinct()

def query_synthetic_object_ephemeris(session, fields, orbit_id=[], name=[], source=[], component=[], night=[], field=[], expnum=[], detnum=[], detname=[]):
    # Synthetic objects have only one fact table linking them to the survey --
    # SyntheticObjectEphemeris -- unlike MPC objects, which split into
    # mpc_observation (actual detections) and mpchecker_ephemeris
    # (predictions). So this resolves to the same join as query_synthetic_objects.
    return query_synthetic_objects(
        session, fields, orbit_id=orbit_id, name=name, source=source, component=component,
        night=night, field=field, expnum=expnum, detnum=detnum, detname=detname,
    )
