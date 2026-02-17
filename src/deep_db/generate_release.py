from .models import SolarSystemObject, Repository, Ephemeris, EphemerisDetectorLocation, Exposure, Night, DetectorExposure
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", type=str, nargs="+")
    parser.add_argument("--datasets", type=str, nargs="+")
    parser.add_argument("--types", type=str, nargs="+")
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements")
    parser.add_argument("--delimiter", type=str, default=",")

    args = parser.parse_args()

    engine = create_engine(args.db, echo=args.echo)
    with Session(engine) as session:
        if args.stats:
            q = session.query(
                SolarSystemObject.name, 
                Night.night,
                func.avg(Ephemeris.v_mag),
                func.count("*")
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
                Night,
                Exposure.night_id == Night.id
            ).group_by(
                SolarSystemObject.name,
                Night.night
            )
        else:
            q = session.query(SolarSystemObject.name).distinct()
            if args.datasets:
                q = q.join(
                    Ephemeris,
                    SolarSystemObject.id == Ephemeris.object_id
                )
            
        if args.types:
            q = q.filter(SolarSystemObject.type.in_(args.types))
        
        if args.datasets:
            q = q.join(EphemerisDetectorLocation, Ephemeris.id == EphemerisDetectorLocation.ephemeris_id)
            q = q.filter(EphemerisDetectorLocation.dataset.in_(args.datasets))
        if args.repos:
            if not args.datasets:
                q = q.join(EphemerisDetectorLocation, Ephemeris.id == EphemerisDetectorLocation.ephemeris_id)
            q = q.join(Repository, EphemerisDetectorLocation.repository_id == Repository.id)
            q = q.filter(Repository.name.in_(args.repos))


        if args.stats:
            print(args.delimiter.join(["name", "night", "mag", "nobs"]))
        for r in q:
            print(args.delimiter.join(map(str, r)))


if __name__ == "__main__":
    main()
