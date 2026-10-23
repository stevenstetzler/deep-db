from sqlalchemy import inspect as sa_inspect
from collections import deque

def _fk_graph(tables, exclude=()):
    """
    Adjacency map {Model: [(OtherModel, on_clause), ...]} built from FK
    constraints. `exclude` is a set of (Model, OtherModel) pairs (either
    order) whose direct FK edge is left out of the graph entirely -- for
    breaking a join-path ambiguity where two tables are connected by more
    than one FK path (e.g. a fact table with both its own exposure_id and
    a detector_exposure_id that itself leads back to the same exposure):
    autojoin's BFS has no notion of "more specific", so without this the
    shortest/first-discovered path wins arbitrarily, which can silently
    pick the looser one.
    """
    from sqlalchemy import and_
    exclude = {pair for a, b in exclude for pair in ((a, b), (b, a))}
    by_table = {t.__table__: t for t in tables}
    edges = {t: [] for t in tables}
    seen = set()
    for t in tables:
        for con in t.__table__.foreign_key_constraints:
            other = by_table.get(con.referred_table)
            if other is None or other is t or (t, other) in seen or (t, other) in exclude:
                continue
            seen.add((t, other))
            seen.add((other, t))
            clause = and_(*(e.parent == e.column for e in con.elements))
            edges[t].append((other, clause))
            edges[other].append((t, clause))
    return edges


def autojoin(q, required, tables, exclude=()):
    """Join every model in `required`, plus whatever intermediates are needed."""
    if not required:
        raise Exception("no fields or filters given -- nothing to query")
    edges = _fk_graph(tables, exclude=exclude)
    required = list(dict.fromkeys(required))  # dedupe, keep order
    root = required[0]
    joined = [root]
    steps = []

    for target in required[1:]:
        if target in joined:
            continue
        # BFS outward from everything already joined
        prev = {j: None for j in joined}
        queue = deque(joined)
        while queue and target not in prev:
            cur = queue.popleft()
            for nxt, clause in edges[cur]:
                if nxt not in prev:
                    prev[nxt] = (cur, clause)
                    queue.append(nxt)
        if target not in prev:
            raise Exception(
                f"cannot join {target.__name__} to {', '.join(t.__name__ for t in joined)}"
            )
        path = []
        node = target
        while prev[node] is not None:
            cur, clause = prev[node]
            path.append((node, clause))
            node = cur
        for node, clause in reversed(path):
            steps.append((node, clause))
            joined.append(node)

    q = q.select_from(root)
    for node, clause in steps:
        q = q.join(node, clause)
    return q

def convert_fields(fields, tables):
    _tables = {
        t.__name__.lower(): t
        for t in tables
    }

    def column_names(table):
        # real mapped columns only; hasattr() also matches things like `metadata`
        return {a.key for a in sa_inspect(table).column_attrs}

    columns = []
    errors = []
    for name in fields:
        # explicit form: Table.column
        if "." in name:
            tname, cname = name.split(".", 1)
            table = _tables.get(tname.lower())
            if table is None:
                errors.append(
                    f"'{name}': unknown table '{tname}' "
                    f"(choose from {', '.join(t.__name__ for t in _tables.values())})"
                )
            elif cname not in column_names(table):
                errors.append(f"'{name}': table {table.__name__} has no column '{cname}'")
            else:
                columns.append(getattr(table, cname))
            continue

        # bare form: search every table
        matches = [t for t in _tables.values() if name in column_names(t)]
        if not matches:
            errors.append(f"'{name}': no table has a column with this name")
        elif len(matches) > 1:
            options = ", ".join(f"{t.__name__}.{name}" for t in matches)
            errors.append(f"'{name}' is ambiguous, use an explicit table name: {options}")
        else:
            columns.append(getattr(matches[0], name))

    if errors:
        raise Exception("\n".join(errors))
    return columns

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
