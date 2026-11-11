#!/usr/bin/env python
"""
One-off backfill for issue #6: wrap synthetic_object_ephemeris.ra into
[0, 360).

Databases ingested before `deep-db ingest fakes` started normalizing RA
hold the DESTNOSIM file's signed (-180, 180] values verbatim. Re-running
the ingest does not fix them (existing rows are skipped), so run this once
against each such database. Safe to re-run: it only touches rows with
ra < 0.

Usage: scripts/backfill/synthetic_object_ephemeris_ra.py sqlite:///deep_database.sqlite
"""
import argparse

from sqlalchemy import create_engine, text


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("url", help="SQLAlchemy database url")
    args = parser.parse_args()

    engine = create_engine(args.url)
    with engine.begin() as conn:
        n = conn.execute(
            text("UPDATE synthetic_object_ephemeris SET ra = ra + 360.0 WHERE ra < 0")
        ).rowcount
        lo, hi = conn.execute(
            text("SELECT min(ra), max(ra) FROM synthetic_object_ephemeris")
        ).one()
    print(f"normalized {n} rows; ra now spans [{lo}, {hi}]")


if __name__ == "__main__":
    main()
