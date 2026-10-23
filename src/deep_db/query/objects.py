from deep_db.query.common import autojoin, convert_fields

def query_known_objects(session, fields, provid=[], permid=[], name=[], night=[], field=[], expnum=[], detnum=[], detname=[]):
    from deep_db.database.mpc import MPCObject, MPCObservation
    from deep_db.database.survey import Night, Field, Exposure, Detector, DetectorExposure
    tables = [MPCObject, MPCObservation, Exposure, Night, Field, Detector, DetectorExposure]
    # (table, condition) for each active filter
    filters = []
    if provid:
        filters.append((MPCObject, MPCObject.provid.in_(provid)))
    if permid:
        filters.append((MPCObject, MPCObject.permid.in_(permid)))
    if name:
        filters.append((MPCObject, MPCObject.name.in_(name)))
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

    # see query_survey's identical exclusion -- ties MPCObservation to the
    # specific DetectorExposure it was taken on instead of every
    # DetectorExposure its Exposure happens to have, whenever Detector/
    # DetectorExposure is involved at all (filtered on, or just selected).
    exclude_edges = []
    if detnum or detname or any(c.class_ in (Detector, DetectorExposure) for c in columns):
        exclude_edges.append((MPCObservation, Exposure))

    required = [c.class_ for c in columns] + [t for t, _ in filters]
    q = autojoin(session.query(*columns), required, tables, exclude=exclude_edges)
    for _, cond in filters:
        q = q.filter(cond)
    return q.distinct()

def query_known_object_ephemeris(session, fields, provid=[], permid=[], name=[], night=[], field=[], expnum=[], detnum=[], detname=[]):
    from deep_db.database.mpc import MPCObject
    from deep_db.database.mpchecker import MPCheckerEphemeris
    from deep_db.database.survey import Night, Field, Exposure, Detector, DetectorExposure
    tables = [MPCObject, MPCheckerEphemeris, Exposure, Night, Field, Detector, DetectorExposure]
    # (table, condition) for each active filter
    filters = []
    if provid:
        filters.append((MPCObject, MPCObject.provid.in_(provid)))
    if permid:
        filters.append((MPCObject, MPCObject.permid.in_(permid)))
    if name:
        filters.append((MPCObject, MPCObject.name.in_(name)))
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

    # see query_survey's identical exclusion -- ties MPCheckerEphemeris to
    # the specific DetectorExposure it was predicted on instead of every
    # DetectorExposure its Exposure happens to have, whenever Detector/
    # DetectorExposure is involved at all (filtered on, or just selected).
    exclude_edges = []
    if detnum or detname or any(c.class_ in (Detector, DetectorExposure) for c in columns):
        exclude_edges.append((MPCheckerEphemeris, Exposure))

    required = [c.class_ for c in columns] + [t for t, _ in filters]
    q = autojoin(session.query(*columns), required, tables, exclude=exclude_edges)
    for _, cond in filters:
        q = q.filter(cond)
    return q.distinct()
