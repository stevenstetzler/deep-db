#!/usr/bin/env python
"""
One-off backfill for issue #7 / PR #12: recompute mpc_tracklet.object_id from
its observations (the single object they agree on, or NULL if they disagree),
instead of whichever observation happened to be upserted last.

Reuses deep_db.ingest.mpc._refresh_tracklet_objects (the function the fixed
ingest path now calls after every upsert) so this can't drift from the fix.

Dry run by default: prints what would change and writes nothing. Pass --apply
to write, in a single transaction.

    python scripts/backfill/mpc_tracklet_object_id.py sqlite:///deep_database.sqlite
    python scripts/backfill/mpc_tracklet_object_id.py sqlite:///deep_database.sqlite --apply
"""
import argparse
from collections import Counter

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from deep_db.database import mpc, mpchecker, survey, synthetic_objects  # noqa: F401 (register ORM relationships)
from deep_db.database.mpc import MPCObservation, MPCTracklet
from deep_db.ingest.mpc import _refresh_tracklet_objects


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", help="SQLAlchemy URL, e.g. sqlite:///deep_database.sqlite")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    engine = create_engine(args.db)
    print("APPLYING CHANGES" if args.apply else "DRY RUN (no changes written; pass --apply)")

    with Session(engine) as session:
        tracklet_ids = [t for (t,) in session.query(MPCTracklet.id)]
        old = dict(session.query(MPCTracklet.id, MPCTracklet.object_id))

        if not args.apply:
            # _refresh_tracklet_objects commits internally, so for a dry run
            # replace commit with flush (keeps the uncommitted transaction
            # visible to the queries below) and roll back everything after.
            session.commit = session.flush

        _refresh_tracklet_objects(session, tracklet_ids)

        new = dict(session.query(MPCTracklet.id, MPCTracklet.object_id))
        changed = [tid for tid in tracklet_ids if old[tid] != new[tid]]
        c = Counter(
            ("NULL" if old[tid] is None else "object") + " -> " + ("NULL" if new[tid] is None else "object")
            for tid in changed
        )
        print(f"\n== mpc_tracklet: {len(tracklet_ids):,} rows, {len(changed):,} change")
        print(f"  NULL before {sum(v is None for v in old.values()):,}, after {sum(v is None for v in new.values()):,}")
        for k, v in sorted(c.items()):
            print(f"    {k:<20}{v:>9,}")

        n_conflict = session.query(func.count()).select_from(
            session.query(MPCObservation.mpc_tracklet_id)
            .filter(MPCObservation.object_id.isnot(None))
            .group_by(MPCObservation.mpc_tracklet_id)
            .having(func.count(func.distinct(MPCObservation.object_id)) > 1)
            .subquery()
        ).scalar()
        print(f"  {n_conflict:,} tracklets have observations attributed to more than one object")

        if args.apply:
            print(f"  updated {len(changed):,} rows")
        else:
            session.rollback()


if __name__ == "__main__":
    main()
