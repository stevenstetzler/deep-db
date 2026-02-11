import os
import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


REMOTE_DB_URL = "https://epyc.astro.washington.edu/~stevengs/DEEP/database/deep_db.db"

def download_db():
    # download sqlite database from URL to local cache location using xdg_dir if available
    import requests
    from pathlib import Path
    import os
    xdg_cache_home = os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")
    cache_dir = Path(xdg_cache_home) / "deep_db"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "deep_db.db"
    response = requests.get(REMOTE_DB_URL + "-SHA", timeout=30)
    response.raise_for_status()
    remote_sha = response.text.strip()
    if db_path.exists():
        local_sha = os.popen(f"sha256sum {db_path}").read().split()[0]
        if local_sha == remote_sha:
            logger.debug("Local database is up to date, skipping download")
            return str(db_path)
    
    logger.debug("Downloading remote database from %s to %s", REMOTE_DB_URL, db_path)
    response = requests.get(REMOTE_DB_URL, timeout=30)
    response.raise_for_status()
    with open(db_path, "wb") as f:
        f.write(response.content)
    return str(db_path)

def get_db():
    import os
    deep_db = os.environ.get("DEEP_DB", None)
    if deep_db is None:
        db_path = download_db()
        deep_db = f"sqlite:///{db_path}"
    return deep_db

def main():
    import argparse
    from sqlalchemy import create_engine, cast, Text
    from sqlalchemy.orm import Session
    from .models import Exposure, Field, Night, SolarSystemObject, Ephemeris, DetectorExposure, Detector, EphemerisSource

    parser = argparse.ArgumentParser(description="Query objects in the DEEP DB")
    parser.add_argument("action", choices=["exposures", "objects"])
    parser.add_argument("--filter", nargs="+", default=[], help="Filter results by a substring (only for fields and objects)")
    parser.add_argument("--query", nargs="+", default=[], help="Columns to query")
    parser.add_argument("--all", action="store_true", help="Return all results")
    parser.add_argument("--log-level", default="INFO", help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    filt = []
    for f in args.filter:
        if "=" in f:
            op = "="
        elif '~' in f:
            op = "~"
        else:
            raise ValueError("Filter must contain '=' or '~' operator")
        k, v = f.split(op)
        if '.' in k:
            k = k.split(".")
        try:
            v = int(v)
        except ValueError:
            pass
        filt.append((k, v, op))

    query = []
    for q in args.query:
        query.append(q.split("."))

    db = get_db()
    logger.debug("Using database at %s", db)
    engine = create_engine(db, echo=False)

    with Session(engine) as session:
        if args.action == "exposures":
            q = session.query(
                Exposure, Field, Night
            ).join(
                Field,
                Exposure.field_id == Field.id
            ).join(
                Night
            ).order_by(
                Exposure.expnum
            )
            for k, v, op in filt:
                if len(k) == 1:
                    if k == "field":
                        if op == "=":
                            q = q.filter(Field.name == v)
                        elif op == "~":
                            q = q.filter(Field.name.regexp_match(v))
                    elif k == "night":
                        if op == "=":
                            q = q.filter(Night.night == int(v))
                        elif op == "~":
                            q = q.filter(cast(Night.night, Text).regexp_match(v))
                else:
                    cls, c = k
                    for i in [Exposure, Field, Night]:
                        if i.__name__ == cls and hasattr(i, c):
                            logger.debug("filtering %s.%s = %s", cls, c, v)
                            if op == "=":
                                q = q.filter(getattr(i, c) == v)
                            elif op == "~":
                                column = getattr(i, c)
                                if not isinstance(column.type, Text):
                                    column = cast(column, Text)
                                q = q.filter(column.regexp_match(v))
            
            logger.debug(str(q))
            default_query = [['Exposure', 'expnum'], ['Exposure', 'band'], ['Exposure', 'exposure'], ['Field', 'name'], ['Night', 'night']] 

            for cls, c in query or default_query:
                print(f"{cls}.{c}", end=" ")
            print()
            if not args.all:
                seen = set()
                for exp, field, night in q:
                    t = []
                    for cls, c in query or default_query:
                        for o in [exp, field, night]:
                            if o.__class__.__name__ == cls and hasattr(o, c):
                                k = f'{cls}.{c}'
                                v = getattr(o, c)
                                t.append(v)
                    t = tuple(t)
                    if t not in seen:
                        print(" ".join(map(str, t)))
                        seen.add(t)
            else:
                for exp, field, night in q:
                    for cls, c in query or default_query:
                        for o in [exp, field, night]:
                            if o.__class__.__name__ == cls and hasattr(o, c):
                                print(getattr(o, c), end=" ")
                    print()
        
        elif args.action == "objects":
            tables = [SolarSystemObject, Ephemeris, Exposure, DetectorExposure, Detector, Field, Night]
            q = session.query(
                *tables
            ).join(
                Ephemeris,
                SolarSystemObject.id == Ephemeris.object_id
            ).join(
                DetectorExposure,
                Ephemeris.detector_exposure_id == DetectorExposure.id
            ).join(
                Exposure,
                DetectorExposure.exposure_id == Exposure.id
            ).join(
                Detector,
                DetectorExposure.detector_id == Detector.id
            ).join(
                Field,
                Exposure.field_id == Field.id
            ).join(
                Night,
                Exposure.night_id == Night.id
            ).join(
                EphemerisSource,
                Ephemeris.source_id == EphemerisSource.id
            )

            q = q.order_by(
                SolarSystemObject.id
            )
            for k, v, op in filt:
                if len(k) == 1:
                    if k == "name":
                        if op == "=":
                            q = q.filter(SolarSystemObject.name == v)
                        elif op == "~":
                            q = q.filter(SolarSystemObject.name.regexp_match(v))
                    elif k == "type":
                        if op == "=":
                            q = q.filter(SolarSystemObject.type == v)
                        elif op == "~":
                            q = q.filter(SolarSystemObject.type.regexp_match(v))
                    elif k == "field":
                        if op == "=":
                            q = q.filter(Field.name == v)
                        elif op == "~":
                            q = q.filter(Field.name.regexp_match(v))
                    elif k == "night":
                        if op == "=":
                            q = q.filter(Night.night == int(v))
                        elif op == "~":
                            q = q.filter(cast(Night.night, Text).regexp_match(v))
                    elif k == "detector":
                        if op == "=":
                            q = q.filter(Detector.number == v)
                        elif op == "~":
                            q = q.filter(cast(Detector.number, Text).regexp_match(v))
                    elif k == "expnum":
                        if op == "=":
                            q = q.filter(Exposure.expnum == int(v))
                        elif op == "~":
                            q = q.filter(cast(Exposure.expnum, Text).regexp_match(v))
                else:
                    cls, c = k
                    for t in tables:
                        if t.__name__ == cls and hasattr(t, c):
                            logger.debug("filtering %s.%s %s %s", cls, c, op, v)
                            if op == "~":
                                q = q.filter(getattr(t, c).regexp_match(v))
                            elif op == "=":
                                q = q.filter(getattr(t, c) == v)
            
            logger.debug(str(q))

            default_query = [['SolarSystemObject', 'name'], ['SolarSystemObject', 'type'], ['Exposure', 'expnum'], ['Ephemeris', 'ra'], ['Ephemeris', 'dec'], ['Field', 'name'], ['Night', 'night'], ['Detector', 'number']]
            for cls, c in query or default_query:
                print(f"{cls}.{c}", end=" ")
            print()
            if not args.all:
                seen = set()
                for result in q:
                    t = []                    
                    for cls, c in query or default_query:
                        for r in result:
                            if r.__class__.__name__ == cls and hasattr(r, c):
                                k = f'{cls}.{c}'
                                v = getattr(r, c)
                                t.append(v)
                    t = tuple(t)
                    if t not in seen:
                        print(" ".join(map(str, t)))
                        seen.add(t)
            else:
                for result in q:
                    for cls, c in query or default_query:
                        for r in result:
                            if r.__class__.__name__ == cls and hasattr(r, c):
                                print(getattr(r, c), end=" ")
                    print()

if __name__ == "__main__":
    main()
