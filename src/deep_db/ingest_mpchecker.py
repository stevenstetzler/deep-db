import sys
from sys import stdin, stderr
from io import StringIO

import astropy.units as u
from joblib import Parallel, delayed
from sqlalchemy import create_engine, insert, tuple_
from sqlalchemy.orm import Session

from .models import (
    hp,
    Base,
    Exposure,
    DetectorExposure,
    SolarSystemObject,
    Ephemeris,
    EphemerisSource,
    Detector,
)
from .ingest_fakes import yield_n_items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_vmag(raw):
    """Parse an 'est. Vmag' field that may be wrapped in square brackets."""
    v = raw.strip()
    if v.startswith("["):
        v = v[1:]
    if v.endswith("]"):
        v = v[:-1]
    return float(v)


def parse_line(line, header):
    """Split a pipe-delimited line into a dict keyed by header. None if malformed."""
    row = line.rstrip("\n").split("|")
    if len(row) != len(header):
        print("WARNING:", row, "is not the same length as", header, file=stderr)
        return None
    return {header[i]: row[i] for i in range(len(header))}


def insert_ignore(session, table, rows):
    """Bulk INSERT ... ON CONFLICT DO NOTHING, dispatched by dialect.

    Relies on the target table's unique constraint to skip duplicates on the
    DB side (fast). Falls back to a plain executemany for dialects that don't
    support ON CONFLICT -- callers de-dupe in-app in that case.
    """
    if not rows:
        return
    dialect = session.bind.dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as _insert
        stmt = _insert(table).on_conflict_do_nothing()
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as _insert
        stmt = _insert(table).on_conflict_do_nothing()
    else:
        stmt = insert(table)
    session.execute(stmt, rows)


def supports_on_conflict(engine):
    return engine.dialect.name in ("postgresql", "sqlite")


# ---------------------------------------------------------------------------
# Worker: process one chunk of lines
# ---------------------------------------------------------------------------

def process_lines(lines, header, db_url, echo):
    engine = create_engine(db_url, echo=echo, pool_pre_ping=True)

    # 1. Parse every line once.
    parsed = []
    for line in lines:
        data = parse_line(line, header)
        if data is None:
            continue
        try:
            parsed.append({
                "packed_des": data["Packed designation"].strip(),
                "expnum": int(data["expnum"]),
                "detector": int(data["detector"]),
                "ra": float(data["ra"]),
                "dec": float(data["dec"]),
                "v_mag": parse_vmag(data["est. Vmag"]),
            })
        except (KeyError, ValueError) as exc:
            print(f"WARNING: skipping row {data}: {exc}", file=stderr)

    if not parsed:
        return

    with Session(engine) as session:
        # 2. Resolve the EphemerisSource once (constant across all rows).
        source = session.query(EphemerisSource).filter_by(name="mpchecker").first()
        if source is None:
            raise RuntimeError("EphemerisSource 'mpchecker' not found in database")
        source_id = source.id

        # 3. Ensure all SolarSystemObjects exist, then map name -> id in bulk.
        names = {p["packed_des"] for p in parsed}
        existing_names = {
            n for (n,) in session.query(SolarSystemObject.name).filter(
                SolarSystemObject.type == "mpchecker",
                SolarSystemObject.name.in_(names),
            )
        }
        missing = names - existing_names
        if missing:
            for n in sorted(missing):
                print(f"missing object {n} in database", file=stderr)
            insert_ignore(
                session,
                SolarSystemObject.__table__,
                [{"name": n, "type": "mpchecker"} for n in missing],
            )
            session.commit()
        obj_map = {
            name: oid
            for oid, name in session.query(
                SolarSystemObject.id, SolarSystemObject.name
            ).filter(
                SolarSystemObject.type == "mpchecker",
                SolarSystemObject.name.in_(names),
            )
        }

        # 4. Resolve all DetectorExposures for this chunk in one query,
        #    keyed by (expnum, detector number).
        expnums = {p["expnum"] for p in parsed}
        detnums = {p["detector"] for p in parsed}
        de_map = {
            (expnum, number): de_id
            for expnum, number, de_id in (
                session.query(Exposure.expnum, Detector.number, DetectorExposure.id)
                .join(Exposure, DetectorExposure.exposure_id == Exposure.id)
                .join(Detector, DetectorExposure.detector_id == Detector.id)
                .filter(Exposure.expnum.in_(expnums))
                .filter(Detector.number.in_(detnums))
            )
        }

        # 5. Build the Ephemeris rows, de-duplicating within the batch on the
        #    unique key (object_id, source_id, detector_exposure_id).
        rows_by_key = {}
        for p in parsed:
            de_id = de_map.get((p["expnum"], p["detector"]))
            if de_id is None:
                print(
                    f"Missing DetectorExposure for expnum {p['expnum']} "
                    f"detector {p['detector']} ra {p['ra']} dec {p['dec']}",
                    file=stderr,
                )
                continue
            object_id = obj_map.get(p["packed_des"])
            if object_id is None:
                print(f"missing object id for {p['packed_des']}", file=stderr)
                continue
            hp_index = int(hp.lonlat_to_healpix(p["ra"] * u.deg, p["dec"] * u.deg))
            key = (object_id, source_id, de_id)
            rows_by_key[key] = {
                "object_id": object_id,
                "source_id": source_id,
                "detector_exposure_id": de_id,
                "ra": p["ra"],
                "dec": p["dec"],
                "v_mag": p["v_mag"],
                "hp_index": hp_index,
            }

        rows = list(rows_by_key.values())
        if not rows:
            session.commit()
            return

        # 6. For dialects without ON CONFLICT support, drop rows that already
        #    exist so the plain insert won't violate the unique constraint.
        if not supports_on_conflict(engine):
            existing_keys = set(
                session.query(
                    Ephemeris.object_id,
                    Ephemeris.source_id,
                    Ephemeris.detector_exposure_id,
                ).filter(
                    Ephemeris.source_id == source_id,
                    tuple_(
                        Ephemeris.object_id, Ephemeris.detector_exposure_id
                    ).in_([(r["object_id"], r["detector_exposure_id"]) for r in rows]),
                )
            )
            rows = [
                r
                for r in rows
                if (r["object_id"], r["source_id"], r["detector_exposure_id"])
                not in existing_keys
            ]

        # 7. Single bulk insert for the whole chunk.
        insert_ignore(session, Ephemeris.__table__, rows)
        session.commit()

    engine.dispose()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest JPL ephemeris data into Deep DB")
    parser.add_argument(
        "mpchecker", nargs="?", default=stdin, type=argparse.FileType("r"),
        help="Path to JPL ephemeris file",
    )
    parser.add_argument("--db-url", required=True, help="Database URL")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements")
    parser.add_argument("--processes", "-J", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=10_000)

    args = parser.parse_args()

    # Buffer stdin so it can be re-read.
    if args.mpchecker is stdin:
        io = StringIO()
        io.write(args.mpchecker.read())
        io.seek(0)
        args.mpchecker = io

    engine = create_engine(args.db_url, echo=args.echo)
    Base.metadata.create_all(engine)

    def gen():
        args.mpchecker.seek(0)
        while line := args.mpchecker.readline():
            yield line

    # ---- Setup phase: source + all objects, committed before the parallel run.
    lines = gen()
    header = [h.strip() for h in next(lines).split("|")]
    assert "Packed designation" in header

    with Session(engine) as session:
        source = session.query(EphemerisSource).filter_by(name="mpchecker").first()
        if source is None:
            session.add(EphemerisSource(name="mpchecker"))
            session.commit()

        objects = set()
        for line in lines:
            data = parse_line(line, header)
            if data:
                objects.add(data["Packed designation"].strip())

        existing = {
            n for (n,) in session.query(SolarSystemObject.name).filter(
                SolarSystemObject.type == "mpchecker",
                SolarSystemObject.name.in_(objects),
            )
        }
        missing = objects - existing
        if missing:
            insert_ignore(
                session,
                SolarSystemObject.__table__,
                [{"name": n, "type": "mpchecker"} for n in sorted(missing)],
            )
            session.commit()

    engine.dispose()

    # ---- Parallel phase: bulk-ingest ephemerides chunk by chunk.
    lines = gen()
    header = [h.strip() for h in next(lines).split("|")]
    Parallel(n_jobs=args.processes)(
        delayed(process_lines)(chunk, header, args.db_url, args.echo)
        for chunk in yield_n_items(lines, args.chunk_size)
    )


if __name__ == "__main__":
    main()