from .models import Base, Exposure, Night, Field, Detector, DetectorExposure, ccdBounds, ccd_num_to_name, ra_dec_to_coordinate
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import astropy.table
from astropy_healpix import HEALPix
import astropy.time
import astropy.units as u

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest data into Deep DB")

    parser.add_argument("exposures")
    parser.add_argument("field_table")
    parser.add_argument("--db-url", required=True, help="Database URL")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements")

    args = parser.parse_args()

    exposures = astropy.table.Table.read(args.exposures)
    field_table = astropy.table.Table.read(args.field_table)

    engine = create_engine(args.db_url, echo=args.echo)
    Base.metadata.create_all(engine)
    
    with Session(engine) as session:
        detector_map = {}
        for detector in range(1, 63):
            det = session.query(Detector).filter_by(number=detector).first()
            if det is not None:
                detector_map[detector] = det.id
                continue
            det = Detector(number=detector)
            session.add(det)
            session.flush()
            detector_map[detector] = det.id

        field_map = {}
        for row in field_table:
            field = session.query(Field).filter_by(name=str(row['field'])).first()
            if field is not None:
                field_map[row['expnum']] = field.id
                continue
            field = Field(name=str(row['field']))
            session.add(field)
            session.flush()
            field_map[row['expnum']] = field.id

        night_map = {}
        for row in exposures:
            night_num = int(row['night'])
            if night_num not in night_map:
                night = session.query(Night).filter_by(night=night_num).first()
                if night is not None:
                    night_map[night_num] = night.id
                    continue
                night = Night(night=night_num)
                session.add(night)
                session.flush()
                night_map[night_num] = night.id

            exposure = session.query(Exposure).filter_by(expnum=int(row['EXPNUM'])).first()
            if exposure is not None:
                print("duplicate expnum:", row['EXPNUM'])
                continue

            obstime = astropy.time.Time(row['mjd'], format='mjd')
            midpoint = obstime + astropy.time.TimeDelta(row['exposure']/2, format='sec')
            ra, dec, hp_index = ra_dec_to_coordinate(row['RA(deg)'] * u.deg, row['DEC(deg)'] * u.deg)
            exposure = Exposure(
                night_id=night_map[night_num],
                field_id=field_map.get(row['EXPNUM'], None),
                ra=ra,
                dec=dec,
                hp_index=hp_index,
                expnum=int(row['EXPNUM']),
                caldat=str(row['caldat']),
                exposure=float(row['exposure']),
                obstime=obstime.to_datetime(),
                midpoint=midpoint.to_datetime(),
                mjd=float(row['mjd']),
                midpoint_mjd=float(row['mjd_midpoint']),
                band=str(row['band']),
                target=str(row['OBJECT']),
            )
            session.add(exposure)
            session.flush()

            for detector in range(1, 63):
                det_exp = session.query(DetectorExposure).filter_by(
                    detector_id=detector_map[detector],
                    exposure_id=exposure.id
                ).first()
                if det_exp is not None:
                    continue
                i = ccd_num_to_name[detector]
                bounds = ccdBounds.get(i, None)
                if bounds is None:
                    continue
                xmin, xmax, ymin, ymax = bounds
                x_center = (xmax + xmin)/2
                y_center = (ymax + ymin)/2
                ra = row['RA(deg)'] + x_center
                dec = row['DEC(deg)'] + y_center
                ra, dec, hp_index = ra_dec_to_coordinate(ra * u.deg, dec * u.deg)

                ra_00 = row['RA(deg)'] + xmin
                dec_00 = row['DEC(deg)'] + ymin
                ra_00, dec_00, hp_index_00 = ra_dec_to_coordinate(ra_00 * u.deg, dec_00 * u.deg)

                ra_01 = row['RA(deg)'] + xmax
                dec_01 = row['DEC(deg)'] + ymin
                ra_01, dec_01, hp_index_01 = ra_dec_to_coordinate(ra_01 * u.deg, dec_01 * u.deg)

                ra_10 = row['RA(deg)'] + xmin
                dec_10 = row['DEC(deg)'] + ymax
                ra_10, dec_10, hp_index_10 = ra_dec_to_coordinate(ra_10 * u.deg, dec_10 * u.deg)

                ra_11 = row['RA(deg)'] + xmax
                dec_11 = row['DEC(deg)'] + ymax
                ra_11, dec_11, hp_index_11 = ra_dec_to_coordinate(ra_11 * u.deg, dec_11 * u.deg)

                det_exp = DetectorExposure(
                    detector_id=detector_map[detector],
                    exposure_id=exposure.id,
                    ra=ra,
                    dec=dec,
                    hp_index=hp_index,
                    ra_00=ra_00,
                    dec_00=dec_00,
                    hp_index_00=hp_index_00,
                    ra_01=ra_01,
                    dec_01=dec_01,
                    hp_index_01=hp_index_01,
                    ra_10=ra_10,
                    dec_10=dec_10,
                    hp_index_10=hp_index_10,
                    ra_11=ra_11,
                    dec_11=dec_11,
                    hp_index_11=hp_index_11
                )
                session.add(det_exp)
                session.flush()
        session.commit()

if __name__ == "__main__":
    main()