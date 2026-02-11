from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sbident import SBIdent
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.jplhorizons import Horizons
import re
from .models import Base, Exposure, SolarSystemObject, Ephemeris, DetectorExposure, ra_dec_to_coordinate


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Database URL")
    parser.add_argument("--location", default="w84", type=str, help="Observatory location code")

    args = parser.parse_args()

    engine = create_engine(args.db)
    Base.metadata.create_all(engine)

    def query_sbident(time, center, expnum):
        r = SBIdent(
            args.location, time, center,
            hwidth=1.25, precision="high"
        ).results
        r['EXPNUM'] = expnum
        r['time'] = time
        return r[['Object name', 'time', 'EXPNUM']]

    with Session(engine) as session:
        for exposure in session.query(Exposure):
            epoch = Time(exposure.midpoint_mjd, format='mjd')
            center = SkyCoord(ra=exposure.ra*u.deg, dec=exposure.dec*u.deg)
            for row in query_sbident(
                epoch,
                center,
                exposure.expnum
            ):
                object_name = row['Object name']

                if (object_name[0] in ['A', 'C', 'P', 'X', 'D'] and object_name[1] == "/"):
                    m = re.compile(r"(.*) \(.*\).*").match(object_name)
                    object_id = m.groups()[0]
                else:
                    m = re.compile(r"(\d+)[ACPXD]/.*").match(object_name)
                    if m:
                        object_id = m.groups(0)
                    else:
                        object_id = re.compile(r".*\((.*)\).*").match(object_name).groups()[0]
                
                obj = session.query(SolarSystemObject).filter_by(name=object_id, type='jpl').first()
                if obj is None and object_id is not None:
                    obj = SolarSystemObject(name=object_id, type='jpl')
                    session.add(obj)
                    session.flush()
    
                ephemeris = session.query(Ephemeris).join(
                    SolarSystemObject, SolarSystemObject.id == Ephemeris.object_id,
                ).join(
                    DetectorExposure, DetectorExposure.id == Ephemeris.detector_exposure_id,
                ).filter(
                    SolarSystemObject.id == obj.id,
                    DetectorExposure.exposure_id == exposure.id,
                ).first()
                if ephemeris is not None:
                    continue

                obj = Horizons(id=object_id, location=args.location, epochs=[epoch.jd])
                ephem = obj.ephemerides()
                ra, dec, hp_index = ra_dec_to_coordinate(ephem['RA'][0] * u.deg, ephem['DEC'][0] * u.deg)
                
                de = session.query(DetectorExposure).filter_by(
                    exposure_id=exposure.id
                ).filter(
                    DetectorExposure.ra_00 < ra,
                    DetectorExposure.ra_11 > ra,
                    DetectorExposure.dec_00 < dec,
                    DetectorExposure.dec_11 > dec,
                ).first()
                if de is None:
                    print(f"Missing DetectorExposure for expnum {exposure.expnum} object {object_id} ra {ephem['RA'][0]} dec {ephem['DEC'][0]}")
                    continue

                ephemeris = Ephemeris(
                    object=obj,
                    detector_exposure=de,
                    ra=ra,
                    dec=dec,
                    hp_index=hp_index,
                    ra_rate=row['RA_rate'],
                    dec_rate=row['DEC_rate'],
                    v_mag=row['V'],
                    alpha=row['alpha'],
                    delta=row['delta'],
                    r=row['r'],
                )
                session.add(ephemeris)
                session.flush()
        session.commit()

if __name__ == "__main__":
    main()
