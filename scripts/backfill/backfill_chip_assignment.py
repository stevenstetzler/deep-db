#!/usr/bin/env python
"""
Backfill for issue #5: recompute DetectorExposure geometry and chip links
with the gnomonic projection (PR #10) on an already-built database.

Steps (each can be skipped):
  1. detector_exposure: recompute the center and corner coordinates and their
     HEALPix indices by inverse gnomonic projection of ccdBounds about each
     exposure's boresight (what ingest_survey() now does).
  2. mpc_observation, mpchecker_ephemeris: recompute detector_exposure_id by
     forward-projecting each row's RA/Dec into its exposure's tangent plane
     (what DetectorExposure.contains() now does).

synthetic_object_ephemeris is NOT touched: its chip assignment comes from the
injection pipeline (CCDNUM), which is the ground truth used in the issue.

Dry run by default: prints what would change and writes nothing. Pass --apply
to write, in a single transaction per table. Work on a COPY of the database
(or take a backup) first.

    python scripts/backfill/backfill_chip_assignment.py sqlite:///deep_database.sqlite
    python scripts/backfill/backfill_chip_assignment.py sqlite:///deep_database.sqlite --apply

Rows that would collide with the mpchecker_ephemeris unique constraint
(object_id, exposure_id, detector_exposure_id) are left unchanged and counted.
"""
import argparse
from collections import Counter

import numpy as np
from astropy.units import deg
from sqlalchemy import create_engine, text

from deep_db.util.coord import gnomonic_to_sky, hp
from deep_db.util.decam import (
    assign_detector_number,
    ccdBounds,
    ccd_num_to_name,
)

LINK_TABLES = [
    # (table, primary key, ra, dec)
    ("mpc_observation", "obsid", "ra", "dec"),
    ("mpchecker_ephemeris", "id", "ra", "dec"),
]
CHUNK = 50_000


def hp_index(ra, dec):
    return hp.lonlat_to_healpix(np.asarray(ra) * deg, np.asarray(dec) * deg).astype(np.int64)


def backfill_geometry(conn, apply):
    rows = conn.execute(text(
        """
        SELECT de.id, d.number, e.ra, e.dec
        FROM detector_exposure de
        JOIN detector d ON d.id = de.detector_id
        JOIN exposure e ON e.id = de.exposure_id
        """
    )).fetchall()
    ids = np.array([r[0] for r in rows])
    nums = np.array([r[1] for r in rows])
    ra0 = np.array([r[2] for r in rows], float)
    dec0 = np.array([r[3] for r in rows], float)
    b = np.array([ccdBounds[ccd_num_to_name[int(n)]] for n in nums])  # xmin xmax ymin ymax

    points = {
        "": ((b[:, 0] + b[:, 1]) / 2, (b[:, 2] + b[:, 3]) / 2),
        "_00": (b[:, 0], b[:, 2]),
        "_01": (b[:, 1], b[:, 2]),
        "_10": (b[:, 0], b[:, 3]),
        "_11": (b[:, 1], b[:, 3]),
    }
    new = {"id": ids}
    for suffix, (x, y) in points.items():
        ra, dec = gnomonic_to_sky(x, y, ra0, dec0)
        new[f"ra{suffix}"], new[f"dec{suffix}"] = ra, dec
        new[f"hp_index{suffix}"] = hp_index(ra, dec)

    old = {
        r[0]: r[1:]
        for r in conn.execute(text("SELECT id, ra_00, dec_00, ra_11, dec_11 FROM detector_exposure"))
    }
    shift = np.array([
        max(abs(new["ra_11"][i] - old[ids[i]][2]) * np.cos(np.radians(new["dec_11"][i])),
            abs(new["dec_11"][i] - old[ids[i]][3])) * 3600
        for i in range(len(ids))
    ])
    print(f"\n== detector_exposure: {len(ids):,} rows")
    print(f"  upper-right corner moves: median {np.median(shift):.1f}\", p90 {np.percentile(shift, 90):.1f}\", max {shift.max():.1f}\"")

    if apply:
        cols = [f"{p}{s}" for s in points for p in ("ra", "dec", "hp_index")]
        stmt = text(f"UPDATE detector_exposure SET {', '.join(f'{c} = :{c}' for c in cols)} WHERE id = :id")
        params = [
            {"id": int(ids[i]), **{c: (int(new[c][i]) if c.startswith("hp") else float(new[c][i])) for c in cols}}
            for i in range(len(ids))
        ]
        conn.execute(stmt, params)
        print(f"  updated {len(params):,} rows")


def backfill_links(conn, apply, table, pk, ra_col, dec_col):
    # (exposure_id, detector number) -> detector_exposure.id
    de_lookup = {
        (e, n): i
        for i, e, n in conn.execute(text(
            "SELECT de.id, de.exposure_id, d.number FROM detector_exposure de JOIN detector d ON d.id = de.detector_id"
        ))
    }
    has_object = table == "mpchecker_ephemeris"
    rows = conn.execute(text(
        f"""
        SELECT t.{pk}, t.detector_exposure_id, t.exposure_id, t.{ra_col}, t.{dec_col}, e.ra, e.dec
               {', t.object_id' if has_object else ''}
        FROM {table} t JOIN exposure e ON e.id = t.exposure_id
        """
    )).fetchall()
    n = len(rows)
    pks = [r[0] for r in rows]
    old = [r[1] for r in rows]
    exp_id = [r[2] for r in rows]
    ra, dec, ra0, dec0 = (np.array([r[k] for r in rows], float) for k in (3, 4, 5, 6))
    detnum = assign_detector_number(ra, dec, ra0, dec0)
    new = [de_lookup.get((exp_id[i], int(detnum[i]))) if detnum[i] else None for i in range(n)]
    on_chip_without_de = sum(1 for i in range(n) if detnum[i] and new[i] is None)

    skipped = 0
    if has_object:
        # NULLs never collide in the unique constraint; only non-null triples matter
        seen = Counter()
        for i in range(n):
            if new[i] is not None and rows[i][7] is not None:
                seen[(rows[i][7], exp_id[i], new[i])] += 1
        for i in range(n):
            if new[i] is not None and rows[i][7] is not None and seen[(rows[i][7], exp_id[i], new[i])] > 1:
                new[i] = old[i]  # keep the existing link
                skipped += 1

    changed = [i for i in range(n) if new[i] != old[i]]
    c = Counter(
        ("NULL" if old[i] is None else "chip") + " -> " + ("NULL" if new[i] is None else "chip")
        for i in changed
    )
    print(f"\n== {table}: {n:,} rows, {len(changed):,} change")
    print(f"  NULL before {sum(o is None for o in old):,}, after {sum(x is None for x in new):,}")
    for k, v in sorted(c.items()):
        print(f"    {k:<14}{v:>9,}")
    if on_chip_without_de:
        print(f"  {on_chip_without_de:,} rows fall on a chip with no detector_exposure row for their exposure (set NULL)")
    if skipped:
        print(f"  {skipped:,} rows left unchanged to avoid violating the unique constraint")

    if apply and changed:
        stmt = text(f"UPDATE {table} SET detector_exposure_id = :de WHERE {pk} = :pk")
        for s in range(0, len(changed), CHUNK):
            conn.execute(stmt, [{"de": new[i], "pk": pks[i]} for i in changed[s:s + CHUNK]])
        print(f"  updated {len(changed):,} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", help="SQLAlchemy URL, e.g. sqlite:///deep_database.sqlite")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--skip-geometry", action="store_true", help="don't recompute detector_exposure corners")
    ap.add_argument("--skip-links", action="store_true", help="don't recompute detector_exposure_id links")
    args = ap.parse_args()

    engine = create_engine(args.db)
    print("APPLYING CHANGES" if args.apply else "DRY RUN (no changes written; pass --apply)")
    # geometry first so the corners are consistent with the links; one transaction
    with engine.begin() as conn:
        if not args.skip_geometry:
            backfill_geometry(conn, args.apply)
        if not args.skip_links:
            for t in LINK_TABLES:
                backfill_links(conn, args.apply, *t)


if __name__ == "__main__":
    main()
