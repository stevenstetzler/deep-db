def _resolve_designations(session, designations):
    """
    Given distinct (unpacked_provisional_designation, permanent_designation,
    name) tuples -- from mpchecker (_upsert_mpchecker_rows) or from the MPC
    database (_upsert_mpc_rows) -- resolve each to an existing
    MPCDesignation.id where one exists, bulk-insert a new MPCDesignation for
    every tuple that doesn't match one, and return the {designation tuple:
    id} lookup covering all of them.

    A tuple is matched against existing designations by whichever
    identifying field(s) it has, in priority order permanent_designation,
    then unpacked_provisional_designation, then name -- not by exact tuple
    equality. This matters most for mpchecker: its "Unpacked Name" never
    gives all three fields at once (a named, numbered object's Unpacked
    Name gives permanent_designation + name but not its
    unpacked_provisional_designation), even though the existing
    MPCDesignation record (built from the fuller MPC identification
    tables) has all three, so exact-tuple equality would treat mpchecker's
    partial tuple as a different (and wrongly duplicated) object.

    A tuple of (None, None, None) -- no identification at all, e.g. an MPC
    observation whose provid/permid/name all came back null from the
    identification joins -- resolves to None directly: no "empty"
    MPCDesignation is created to represent "no designation" (MPCDesignation
    requires unpacked_provisional_designation); callers should leave
    object_id null on those rows instead.

    by_permanent/by_provisional/by_name are seeded from the existing
    MPCObject rows, then updated as each new one is decided within the
    loop below -- not just at the end -- so that two designations in the
    *same* call that share an identifying field (e.g. mpchecker rows from
    different exposures naming the same numbered object, which never get
    a full (provid, permid, name) tuple at once) resolve to one shared
    new MPCObject instead of each inserting their own duplicate.
    """
    from deep_db.database.mpc import MPCObject

    by_permanent = {}
    by_provisional = {}
    by_name = {}
    for d in session.query(MPCObject):
        if d.permid is not None:
            by_permanent[d.permid] = d
        if d.provid is not None:
            by_provisional[d.provid] = d
        if d.name is not None:
            by_name[d.name] = d

    def resolve(designation):
        provid, permanent, name = designation
        if permanent is not None and permanent in by_permanent:
            return by_permanent[permanent]
        if provid is not None and provid in by_provisional:
            return by_provisional[provid]
        if name is not None and name in by_name:
            return by_name[name]
        return None

    objects_by_designation = {}
    new_objects = []
    for designation in designations:
        if designation == (None, None, None):
            objects_by_designation[designation] = None
            continue
        existing = resolve(designation)
        if existing is not None:
            objects_by_designation[designation] = existing
            continue

        provid, permanent, name = designation
        obj = MPCObject(provid=provid, permid=permanent, name=name)
        new_objects.append(obj)
        objects_by_designation[designation] = obj
        if provid is not None:
            by_provisional[provid] = obj
        if permanent is not None:
            by_permanent[permanent] = obj
        if name is not None:
            by_name[name] = obj

    if new_objects:
        session.add_all(new_objects)
        session.flush()  # populate ids for newly created designations
        session.commit()

    return {
        designation: (obj.id if obj is not None else None)
        for designation, obj in objects_by_designation.items()
    }
