#!/usr/bin/env python
"""
Before/after check of chip (detector_exposure) assignment for issue #5.

"Before" is the detector_exposure_id stored in the database (assigned with the
old flat RA/Dec boxes). "After" re-derives the chip for every row with the
gnomonic projection used by DetectorExposure.contains() (ccdBounds in the
tangent plane about the exposure boresight).

The database is opened read-only; nothing is modified.

    python scripts/check_chip_assignment.py deep_database.sqlite
"""
import argparse
import sqlite3
from collections import Counter

import numpy as np

from deep_db.util.coord import sky_to_gnomonic
from deep_db.util.decam import assign_detector_number, ccdBounds, ccd_name_to_num

# (table, primary key column, ra column, dec column)
TABLES = [
    ("mpc_observation", "obsid", "ra", "dec"),
    ("mpchecker_ephemeris", "id", "ra", "dec"),
]

CHIP_NUMS = np.array(sorted(ccd_name_to_num.values()))
_num_to_name = {v: k for k, v in ccd_name_to_num.items()}
BOUNDS = np.array([ccdBounds[_num_to_name[n]] for n in CHIP_NUMS])  # (62, 4), edge-distance calc below


def report(con, table, pk, ra_col, dec_col):
    cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    missing = {pk, "exposure_id", "detector_exposure_id", ra_col, dec_col} - cols
    if missing:
        print(f"\n== {table}: skipped, missing columns {sorted(missing)}")
        return

    rows = con.execute(
        f"""
        SELECT t.{pk}, t.detector_exposure_id, t.{ra_col}, t.{dec_col},
               e.id, e.ra, e.dec, d.number
        FROM {table} t
        JOIN exposure e ON e.id = t.exposure_id
        LEFT JOIN detector_exposure de ON de.id = t.detector_exposure_id
        LEFT JOIN detector d ON d.id = de.detector_id
        """
    ).fetchall()
    n = len(rows)
    stored_de = np.array([r[1] if r[1] is not None else -1 for r in rows])
    ra = np.array([r[2] for r in rows], float)
    dec = np.array([r[3] for r in rows], float)
    exp_id = np.array([r[4] for r in rows])
    ra0 = np.array([r[5] for r in rows], float)
    dec0 = np.array([r[6] for r in rows], float)
    old_det = np.array([r[7] if r[7] is not None else 0 for r in rows])

    new_det = assign_detector_number(ra, dec, ra0, dec0)

    old_null, new_null = old_det == 0, new_det == 0
    print(f"\n== {table}: {n:,} rows")
    print(f"  NULL chip before: {old_null.sum():>9,} ({100 * old_null.mean():.2f}%)")
    print(f"  NULL chip after : {new_null.sum():>9,} ({100 * new_null.mean():.2f}%)")

    trans = Counter()
    trans["NULL -> chip"] = int((old_null & ~new_null).sum())
    trans["NULL -> NULL"] = int((old_null & new_null).sum())
    trans["chip -> NULL"] = int((~old_null & new_null).sum())
    both = ~old_null & ~new_null
    trans["chip -> same chip"] = int((both & (old_det == new_det)).sum())
    trans["chip -> different chip"] = int((both & (old_det != new_det)).sum())
    print("  transitions:")
    for k, v in trans.items():
        print(f"    {k:<24}{v:>9,}  ({100 * v / n:.2f}%)")
    if old_null.any():
        print(
            f"  of the {old_null.sum():,} NULL-chip rows, "
            f"{100 * trans['NULL -> chip'] / old_null.sum():.1f}% land on a chip under the gnomonic test"
        )
    if both.any() or trans["chip -> NULL"]:
        linked = ~old_null
        print(
            f"  of the {linked.sum():,} already-linked rows, "
            f"{100 * (linked & ~new_null).sum() / linked.sum():.1f}% still land on a chip, "
            f"{100 * (both & (old_det == new_det)).sum() / linked.sum():.1f}% on the same chip"
        )

    # NULL rows that stay NULL: how far are they from the nearest chip edge?
    still = old_null & new_null
    if still.any():
        xi, eta = sky_to_gnomonic(ra[still], dec[still], ra0[still], dec0[still])
        dx = np.maximum(BOUNDS[:, 0] - xi[:, None], xi[:, None] - BOUNDS[:, 1]).clip(min=0)
        dy = np.maximum(BOUNDS[:, 2] - eta[:, None], eta[:, None] - BOUNDS[:, 3]).clip(min=0)
        dist = np.hypot(dx, dy).min(axis=1) * 3600
        print(
            f"  still-NULL rows: distance to nearest chip edge (arcsec) "
            f"median {np.median(dist):.1f}, p90 {np.percentile(dist, 90):.1f}, max {dist.max():.1f}"
        )

    # distinct objects (mpc_observation only): trkids with only unlinked obs
    if table == "mpc_observation" and "mpc_tracklet_id" in cols:
        trk = np.array([r[0] for r in con.execute(f"SELECT mpc_tracklet_id FROM {table} t JOIN exposure e ON e.id = t.exposure_id")])
        if len(trk) == n:
            for label, null in (("before", old_null), ("after", new_null)):
                only_null = set(np.unique(trk[null])) - set(np.unique(trk[~null]))
                print(f"  tracklets with only unlinked observations, {label}: {len(only_null):,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", help="path to deep_database.sqlite")
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    for t in TABLES:
        report(con, *t)


if __name__ == "__main__":
    main()
