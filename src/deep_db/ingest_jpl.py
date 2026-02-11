import astropy_healpix
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import astropy.units as u
from joblib import Parallel, delayed
from .models import hp, Base, Exposure, DetectorExposure, SolarSystemObject, Ephemeris, EphemerisSource
from .ingest_fakes import yield_n_items


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest JPL ephemeris data into Deep DB")
    parser.add_argument("jpl_ephemeris", type=argparse.FileType('r'), help="Path to JPL ephemeris file")
    parser.add_argument("--db-url", required=True, help="Database URL")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements")
    parser.add_argument("--processes", "-J", type=int, default=16)

    args = parser.parse_args()

    engine = create_engine(args.db_url, echo=args.echo)
    Base.metadata.create_all(engine)

    def process_rows(rows):
        engine = create_engine(args.db_url, echo=args.echo)
        with Session(engine) as session:
            for row in rows:
                # print(row)
                parts = row.strip().split("|")
                if len(parts) != 12:
                    continue
                object_name, _, expnum, time, ra, dec, ra_rate, dec_rate, v, alpha, delta, r = parts
                expnum = int(expnum)
                time = float(time)
                ra = float(ra)
                dec = float(dec)
                ra_rate = float(ra_rate)
                dec_rate = float(dec_rate)
                v = float(v)
                alpha = float(alpha)
                delta = float(delta)
                r = float(r)
                hp_index = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)

                obj = session.query(SolarSystemObject).filter_by(name=object_name).first()
                if obj is None:
                    obj = SolarSystemObject(name=object_name, type="sbident")
                    session.add(obj)
                
                neighbors_of_neighbors = list(map(int, sum(map(lambda x : list(hp.neighbours(x)), hp.neighbours(hp_index)), [])))
                print(expnum, hp_index, neighbors_of_neighbors, file=sys.stderr)
                de = session.query(
                    DetectorExposure
                ).join(
                    Exposure, DetectorExposure.exposure_id == Exposure.id
                ).filter(
                    Exposure.expnum == int(expnum)
                ).filter(
                    DetectorExposure.hp_index.in_(neighbors_of_neighbors) # is this necessary? yes, because hp_index refers to the center of the detector
                ).filter(
                    DetectorExposure.ra_00 < float(ra),
                    DetectorExposure.ra_11 > float(ra),
                    DetectorExposure.dec_00 < float(dec),
                    DetectorExposure.dec_11 > float(dec),
                ).first()
                
                if de is None:
                    print(f"Missing DetectorExposure for expnum {expnum} hp_index {hp_index} ra {ra} dec {dec}", file=sys.stderr)
                    continue
                
                ephemeris_source = session.query(EphemerisSource).filter_by(name="Horizons").first()
                if ephemeris_source is None:
                    raise RuntimeError("EphemerisSource 'Horizons' not found in database")

                eph = session.query(Ephemeris).filter_by(
                    object=obj,
                    detector_exposure=de,
                    source=ephemeris_source,
                ).first()
                if eph is None:
                    eph = Ephemeris(
                        object=obj,
                        source=ephemeris_source,
                        detector_exposure=de,
                        ra=ra,
                        dec=dec,
                        ra_rate=ra_rate,
                        dec_rate=dec_rate,
                        v_mag=v,
                        alpha=alpha,
                        delta=delta,
                        r=r,
                        hp_index=int(hp_index)
                    )
                    session.add(eph)

                session.commit()

    def gen():
        while line := args.jpl_ephemeris.readline():
            yield line

    def gen_chunks(nchunks, nlines):
        chunks = []
        for chunk in yield_n_items(gen(), nlines):
            chunk.append(chunk)
            if len(chunks) == nchunks:
                yield chunks
                chunks = []
        if chunks:
            yield chunks

    engine = create_engine(args.db_url, echo=args.echo)
    with Session(engine) as session:
        ephemeris_source = session.query(EphemerisSource).filter_by(name="Horizons").first()
        if ephemeris_source is None:
            ephemeris_source = EphemerisSource(name="Horizons")
            session.add(ephemeris_source)
            session.commit()
        
    chunk_size = 10_000
    Parallel(n_jobs=args.processes)(delayed(process_rows)(chunk) for chunk in yield_n_items(gen(), chunk_size))


if __name__ == "__main__":
    main()
