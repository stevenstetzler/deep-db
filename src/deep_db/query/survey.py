from deep_db.query.common import autojoin, convert_fields

_OBJECT_JOINS = ("mpc_observation", "mpchecker_ephemeris", "synthetic_object_ephemeris")

def query_survey(session, fields, night=[], field=[], expnum=[], detnum=[], detname=[], provid=[], permid=[], object_name=[], join=None):
    from deep_db.database.survey import Night, Field, Exposure, Detector, DetectorExposure
    tables = [Night, Field, Exposure, Detector, DetectorExposure]
    # (table, condition) for each active filter
    filters = []
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

    if join is not None and join not in _OBJECT_JOINS:
        raise Exception(f"join must be one of {_OBJECT_JOINS}, got {join!r}")
    if (provid or permid or object_name) and join is None:
        raise Exception(
            f"provid/permid/object_name require join to be one of {_OBJECT_JOINS}, "
            "to say which table to reach the object through"
        )
    if join == "synthetic_object_ephemeris" and (provid or permid):
        raise Exception("synthetic objects have no provid/permid -- filter on object_name instead")

    fact_table = None
    if join == "mpc_observation":
        from deep_db.database.mpc import MPCObservation, MPCObject
        tables += [MPCObservation, MPCObject]
        fact_table, object_table = MPCObservation, MPCObject
    elif join == "mpchecker_ephemeris":
        from deep_db.database.mpchecker import MPCheckerEphemeris
        from deep_db.database.mpc import MPCObject
        tables += [MPCheckerEphemeris, MPCObject]
        fact_table, object_table = MPCheckerEphemeris, MPCObject
    elif join == "synthetic_object_ephemeris":
        from deep_db.database.synthetic_objects import SyntheticObjectEphemeris, SyntheticObject, SyntheticObjectDesignation
        tables += [SyntheticObjectEphemeris, SyntheticObject, SyntheticObjectDesignation]
        fact_table, object_table = SyntheticObjectEphemeris, SyntheticObjectDesignation

    if provid:
        filters.append((object_table, object_table.provid.in_(provid)))
    if permid:
        filters.append((object_table, object_table.permid.in_(permid)))
    if object_name:
        filters.append((object_table, object_table.name.in_(object_name)))

    columns = convert_fields(fields, tables)

    # When Detector/DetectorExposure is involved at all -- filtered on, or
    # just selected as an output column -- tie the fact table to that
    # specific DetectorExposure (via its own detector_exposure_id) rather
    # than letting autojoin take the shorter path straight to Exposure,
    # which would join every DetectorExposure belonging to the exposure,
    # not just the one the object was actually observed/predicted on.
    exclude_edges = []
    if fact_table is not None and (
        detnum or detname or any(c.class_ in (Detector, DetectorExposure) for c in columns)
    ):
        exclude_edges.append((fact_table, Exposure))

    required = [c.class_ for c in columns] + [t for t, _ in filters]
    q = autojoin(session.query(*columns), required, tables, exclude=exclude_edges)
    for _, cond in filters:
        q = q.filter(cond)
    return q.distinct()
