from .models import Base, Detector, Exposure, DetectorExposure, SolarSystemObject, FakeLightCurveProperties, FakeBinaryProperties, Ephemeris, Orbit, CartesianState, KeplerianState, ra_dec_to_coordinate, EphemerisSource
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import astropy.units as u
from pathlib import Path
import lsst.daf.butler as dafButler
from joblib import Parallel, delayed
import astropy.table
import sys
from tqdm import tqdm

def yield_n_items(iterable, n, filt=lambda x : True):
    """
    Yields successive N-sized chunks (lists) from an iterable.

    Args:
        iterable: The input iterable (e.g., a list, set, or generator).
        n: The maximum number of items to yield in each chunk.

    Yields:
        A list containing up to N items from the iterable.
    """
    if n <= 0:
        raise ValueError("N must be greater than zero.")
        
    chunk = []
    for item in filter(filt, iterable):
        chunk.append(item)
        # Check if the chunk has reached the desired size
        if len(chunk) == n:
            yield chunk
            chunk = []
            
    # Yield the last remaining chunk if it's not empty
    if chunk:
        yield chunk

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest data into Deep DB")

    parser.add_argument("fakes_dir", type=Path)
    parser.add_argument("repo", type=str)
    parser.add_argument("--dataset", type=str, default="injected_postISRCCD_catalog")
    parser.add_argument("--collections", type=str, default="DEEP/*")
    parser.add_argument("--db-url", required=True, help="Database URL")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements")
    parser.add_argument("--processes", "-J", type=int, default=1)

    args = parser.parse_args()

    engine = create_engine(args.db_url, echo=args.echo)
    Base.metadata.create_all(engine)

    # asteroid_population_states.fits
    # tno_population_states.fits
    # asteroid_ephem.fits
    # tno_ephem_with_binaries.fits

    def ingest_fakes(rows):
        engine = create_engine(args.db_url, echo=args.echo)
        with Session(engine) as session:
            for row in rows:
                cartesian_state = CartesianState(
                    x=float(row['xv'][0]),
                    y=float(row['xv'][1]),
                    z=float(row['xv'][2]),
                    vx=float(row['xv'][3]),
                    vy=float(row['xv'][4]),
                    vz=float(row['xv'][5])
                )
                keplerian_state = KeplerianState(
                    a=float(row['aei'][0]),
                    e=float(row['aei'][1]),
                    i=float(row['aei'][2]),
                    omega=float(row['aei'][3]),
                    w=float(row['aei'][4]),
                    M=float(row['aei'][5])
                )
                orbit = Orbit(
                    cartesian_state=cartesian_state,
                    keplerian_state=keplerian_state,
                    epoch=0.0,
                )
                

                if row['ORBITID'] in binary_orbitids and 't' in row['object_name']:
                    for suffix in ["1", "2"]: # for each object in the binary
                        obj = session.query(SolarSystemObject).filter_by(name=row['object_name'] + f"_{suffix}", type='fake').first()
                        if obj is None:
                            obj = SolarSystemObject(name=row['object_name'] + f"_{suffix}", type='fake', orbits=[orbit])
                            session.add(obj)
                            session.flush()
                else:
                    obj = session.query(SolarSystemObject).filter_by(name=row['object_name'], type='fake').first()
                    if obj is None:
                        obj = SolarSystemObject(name=row['object_name'], type='fake', orbits=[orbit])
                        session.add(obj)
                        session.flush()
            session.commit()

    def ingest_binary_properties(rows):
        engine = create_engine(args.db_url, echo=args.echo)
        with Session(engine) as session:
            for row in rows:
                delta_h = row['DELTA_H']
                separation = row['SEPARATION']
                angle = row['ANGLE']

                fakes = session.query(SolarSystemObject).filter(
                    SolarSystemObject.name.contains('t' + str(row['ORBITID']))
                ).all()
                if len(fakes) == 0:
                    print(f"Missing SolarSystemObject for binary {row['ORBITID']}", file=sys.stderr)
                    continue

                binary_fake_props = session.query(FakeBinaryProperties).join(
                    FakeBinaryProperties.objects
                ).filter(
                    SolarSystemObject.name.contains('t' + str(row['ORBITID']))
                ).first()
                if binary_fake_props is not None:
                    continue

                binary_fake_props = FakeBinaryProperties(
                    delta_h=float(delta_h),
                    separation=float(separation),
                    angle=float(angle),
                    objects=list(fakes),
                )
                session.add(binary_fake_props)
                session.flush()
            session.commit()

    def ingest_light_curve_properties(t):
        engine = create_engine(args.db_url, echo=args.echo)
        with Session(engine) as session:
            for row in t:
                h_vr = row['H_VR']
                amp = row['AMP']
                period = row['PERIOD']
                phase = row['PHASE']

                fake = session.query(SolarSystemObject).filter_by(name=row['object_name']).first()
                if fake is None:
                    print(f"Missing SolarSystemObject for fake {row['object_name']}")
                    continue

                fake_props = session.query(FakeLightCurveProperties).join(
                    FakeLightCurveProperties.objects
                ).filter(
                    SolarSystemObject.name == row['object_name']
                ).first()
                
                if fake_props is None:
                    fake_props = FakeLightCurveProperties(
                        objects=[fake],
                    )

                fake_props.h_vr = float(h_vr)
                fake_props.amp = float(amp)
                fake_props.period = float(period)
                fake_props.phase = float(phase)
                session.add(fake_props)
                session.flush()
            session.commit()


    def ingest_ephemeris_from_table(t):
        engine = create_engine(args.db_url, echo=args.echo)
        with Session(engine) as session:
            for row in t:
                if 't' in row['object_name'] and row['ORBITID'] in binary_orbitids:
                    print(f"Skipping binary ephemeris for {row['object_name']} in ingest_ephemeris_from_table", file=sys.stderr)
                    continue

                ra, dec, hp_index = ra_dec_to_coordinate(float(row['RA']) * u.deg, float(row['DEC']) * u.deg)

                if "DETECTOR" in row.colnames:
                    detector_exposure = session.query(DetectorExposure).join(
                        Exposure, Exposure.id == DetectorExposure.exposure_id
                    ).join(
                        Detector, Detector.id == DetectorExposure.detector_id
                    ).filter(
                        Exposure.expnum == int(row['EXPNUM']),
                        Detector.number == int(row['DETECTOR'])
                    ).first()
                    if detector_exposure is None:
                        print(f"Missing DetectorExposure for object {row['object_name']} expnum {row['EXPNUM']} detector {row['DETECTOR']}", file=sys.stderr)
                        continue
                else:
                    detector_exposure = session.query(DetectorExposure).join(
                        Exposure, Exposure.id == DetectorExposure.exposure_id
                    ).filter(
                        Exposure.expnum == int(row['EXPNUM']),
                    ).filter(
                        DetectorExposure.ra_00 < float(ra),
                        DetectorExposure.ra_11 > float(ra),
                        DetectorExposure.dec_00 < float(dec),
                        DetectorExposure.dec_11 > float(dec),
                    ).first()
                    
                    if detector_exposure is None:
                        print(f"Missing DetectorExposure for object {row['object_name']} expnum {row['EXPNUM']} ra {ra} dec {dec}", file=sys.stderr)
                        continue

                # how do I do this for binaries?
                fake = session.query(SolarSystemObject).filter(
                    SolarSystemObject.name.contains(row['object_name'])
                ).first()
                if fake is None:
                    print(f"Missing SolarSystemObject for fake {row['object_name']}", file=sys.stderr)
                    continue
                
                orbit = session.query(Orbit).join(
                    Orbit.objects
                ).filter(
                    SolarSystemObject.name.contains(row['object_name'])
                ).first()

                ephem_source = session.query(EphemerisSource).filter_by(name=str(orbit.id)).first()
                if ephem_source is None:
                    print(f"Missing EphemerisSource for orbit {orbit.id}", file=sys.stderr)
                    continue

                ephem = session.query(Ephemeris).filter_by(
                    source=ephem_source,
                    object=fake,
                    detector_exposure=detector_exposure,
                ).first()
                
                if ephem is None:
                    ephem = Ephemeris(
                        source=ephem_source,
                        object=fake,
                        detector_exposure=detector_exposure,
                    )
                
                ephem.ra = ra
                ephem.dec = dec
                ephem.hp_index = hp_index
                ephem.v_mag = float(row['MAG'])
                ephem.r = float(row['r'])
                ephem.delta = float(row['d'])

                session.add(ephem)
                session.flush()
            session.commit()

    def ingest_ephemeris_from_butler(ref):
        engine = create_engine(args.db_url, echo=args.echo)
        _butler = dafButler.Butler(args.repo)
        butler = dafButler.Butler(
            args.repo,
            collections=_butler.registry.queryCollections(args.collections)
        )
        with Session(engine) as session:
            for row in butler.get(ref):
                orbitid = row['ORBITID']
                fake = session.query(SolarSystemObject).filter_by(name=str(orbitid)).first()
                if fake is None:
                    print(f"Missing SolarSystemObject for fake {orbitid}", file=sys.stderr)
                    continue

                detector_exposure = session.query(DetectorExposure).join(
                    Exposure, Exposure.id == DetectorExposure.exposure_id
                ).join(
                    Detector, Detector.id == DetectorExposure.detector_id
                ).filter(
                    Exposure.expnum == int(ref.dataId['exposure']),
                    Detector.number == int(ref.dataId['detector'])
                ).first()
                if detector_exposure is None:
                    print(f"Missing DetectorExposure for expnum {ref.dataId['exposure']} detector {ref.dataId['detector']}", file=sys.stderr)
                    continue

                # binaries break the assumption that there is one ephemeris per object per detector exposure, instead there will be two
                # one solution is to modify the object_id for binaries to discriminate them (e.g. add 1e7 to one and 2e7 to the other)
                ephem = session.query(Ephemeris).filter_by(
                    object_id=fake.id,
                    detector_exposure_id=detector_exposure.id,
                ).first()
                if ephem is None:
                    ephem = Ephemeris(
                        object=fake,
                        detector_exposure=detector_exposure,
                    )

                ra, dec, hp_index = ra_dec_to_coordinate(float(row['ra']) * u.deg, float(row['dec']) * u.deg)
                ephem.ra = ra
                ephem.dec = dec
                ephem.hp_index = hp_index
                ephem.v_mag = float(row['mag'])
                # ephem.r = float(row['r'])
                # ephem.delta = float(row['d'])

                session.add(ephem)
                session.flush()
            session.commit()

    binary_properties = astropy.table.Table.read(args.fakes_dir / "binary_properties.fits")
    binary_properties['object_name'] = list(map(lambda x : 't' + str(x), binary_properties['ORBITID']))
    binary_orbitids = set(binary_properties['ORBITID'])
    chunk_size = 10_000

    # # ingest population states
    # for f in ['asteroid_population_states.fits', 'tno_population_states.fits']:
    #     print("opening", f, file=sys.stderr)
    #     table = astropy.table.Table.read(args.fakes_dir / f)
    #     if 'asteroid' in f:
    #         table['object_name'] = list(map(lambda x : 'a' + str(x), table['ORBITID']))
    #     else:
    #         table['object_name'] = list(map(lambda x : 't' + str(x), table['ORBITID']))
    #     total = len(table)
    #     print("ingesting", total, "orbits from", f, file=sys.stderr)

    #     Parallel(n_jobs=args.processes)(
    #         delayed(ingest_fakes)(
    #             chunk
    #         )
    #         for chunk in tqdm(yield_n_items(table, chunk_size), total=int(total/chunk_size + 0.5))
    #     )
    
    # chunk_size = 1_000

    # print("ingesting", len(binary_properties), "binary properties", file=sys.stderr)
    # # ingest binary properties
    # Parallel(n_jobs=args.processes)(
    #     delayed(ingest_binary_properties)(
    #         chunk
    #     )
    #     for chunk in tqdm(yield_n_items(binary_properties, chunk_size), total=int(len(binary_properties)/chunk_size + 0.5))
    # )

    # chunk_size = 10_000
    # # ingest light curve properties
    # for f in ['asteroid_properties.fits', 'tno_properties.fits']:
    #     print("opening", f, file=sys.stderr)
    #     table = astropy.table.Table.read(args.fakes_dir / f)
    #     if 'asteroid' in f:
    #         table['object_name'] = list(map(lambda x : 'a' + str(x), table['ORBITID']))
    #     else:
    #         table['object_name'] = list(map(lambda x : 't' + str(x), table['ORBITID']))
    #     total = len(table)
    #     print("ingesting", total, "light curve properties from", f, file=sys.stderr)

    #     Parallel(n_jobs=args.processes)(
    #         delayed(ingest_light_curve_properties)(
    #             chunk
    #         )
    #         for chunk in tqdm(yield_n_items(table, chunk_size), total=int(total/chunk_size + 0.5))
    #     )
        
    # ingest ephem
    chunk_size = 10_000
    for f in ['asteroid_ephem.fits', 'tno_ephem_with_binaries.fits']:
        table = astropy.table.Table.read(args.fakes_dir / f)
        if 'asteroid' in f:
            table['object_name'] = list(map(lambda x : 'a' + str(x), table['ORBITID']))
        else:
            table['object_name'] = list(map(lambda x : 't' + str(x), table['ORBITID']))
        total = len(table)
        
        # Add EphemerisSources for all orbits first
        print("adding EphemerisSources for orbits", file=sys.stderr)

        seen = set()
        with Session(engine) as session:
            for row in table:
                if row['object_name'] in seen:
                    continue
                seen.add(row['object_name'])

                fake = session.query(SolarSystemObject).filter(
                    SolarSystemObject.name == row['object_name']
                ).first()
                if fake is None:
                    print(f"Missing SolarSystemObject for fake {row['object_name']}", file=sys.stderr)
                    continue
                
                orbit = session.query(Orbit).join(
                    Orbit.objects
                ).filter(
                    SolarSystemObject.name == row['object_name']
                ).first()
                ephem_source = session.query(EphemerisSource).filter_by(name=str(orbit.id)).first()
                if ephem_source is None:
                    ephem_source = EphemerisSource(name=str(orbit.id))
                    session.add(ephem_source)
                    session.flush()
            session.commit()

        print("ingesting", total, f"ephemerides from {f}", file=sys.stderr)
        Parallel(n_jobs=args.processes)(
            delayed(ingest_ephemeris_from_table)(
                chunk
            )
            for chunk in tqdm(yield_n_items(table, chunk_size), total=int(total/chunk_size + 0.5))
        )
        
if __name__ == "__main__":
    main()