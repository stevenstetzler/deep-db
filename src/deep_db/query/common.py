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
