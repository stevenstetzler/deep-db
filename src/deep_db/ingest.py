from .models import Base, Exposure, Night, Field, Detector, DetectorExposure, ccdBounds, ccd_num_to_name, ra_dec_to_coordinate_array
from sqlalchemy import create_engine, select, insert
from sqlalchemy.orm import Session
import astropy.table
from astropy_healpix import HEALPix
import astropy.time
import astropy.units as u
import numpy as np

CHUNK_SIZE = 10000


def _chunked(rows, size=CHUNK_SIZE):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


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
        # --- Detectors: bulk create the fixed set of 1-62 detectors ---
        existing_detectors = dict(
            session.execute(select(Detector.number, Detector.id)).all()
        )
        missing_detector_numbers = [
            n for n in range(1, 63) if n not in existing_detectors
        ]
        if missing_detector_numbers:
            session.execute(
                insert(Detector),
                [{"number": n} for n in missing_detector_numbers],
            )
            session.commit()
        detector_map = dict(
            session.execute(select(Detector.number, Detector.id)).all()
        )

        # --- Fields: bulk create distinct field names, then map expnum -> field_id ---
        field_names = sorted({str(row['field']) for row in field_table})
        existing_fields = dict(
            session.execute(select(Field.name, Field.id)).all()
        )
        missing_field_names = [n for n in field_names if n not in existing_fields]
        if missing_field_names:
            session.execute(
                insert(Field),
                [{"name": n} for n in missing_field_names],
            )
            session.commit()
        field_name_to_id = dict(
            session.execute(select(Field.name, Field.id)).all()
        )
        field_map = {
            int(row['expnum']): field_name_to_id[str(row['field'])]
            for row in field_table
        }

        # --- Nights: bulk create distinct nights, then map night -> id ---
        night_numbers = sorted({int(row['night']) for row in exposures})
        existing_nights = dict(
            session.execute(select(Night.night, Night.id)).all()
        )
        missing_nights = [n for n in night_numbers if n not in existing_nights]
        if missing_nights:
            session.execute(
                insert(Night),
                [{"night": n} for n in missing_nights],
            )
            session.commit()
        night_map = dict(
            session.execute(select(Night.night, Night.id)).all()
        )

        # --- Exposures: bulk insert new rows, skipping already-ingested expnums ---
        existing_expnums = set(
            session.execute(select(Exposure.expnum)).scalars().all()
        )

        candidate_rows = [
            row for row in exposures if int(row['EXPNUM']) not in existing_expnums
        ]

        # de-duplicate expnums within the input file itself
        seen = set()
        new_exposure_rows = []
        for row in candidate_rows:
            expnum = int(row['EXPNUM'])
            if expnum in seen:
                print("duplicate expnum:", expnum)
                continue
            seen.add(expnum)
            new_exposure_rows.append(row)

        if new_exposure_rows:
            ra_arr = np.array([float(row['RA(deg)']) for row in new_exposure_rows]) * u.deg
            dec_arr = np.array([float(row['DEC(deg)']) for row in new_exposure_rows]) * u.deg
            ra_deg, dec_deg, hp_index = ra_dec_to_coordinate_array(ra_arr, dec_arr)

            mjd_arr = np.array([float(row['mjd']) for row in new_exposure_rows])
            exptime_arr = np.array([float(row['exposure']) for row in new_exposure_rows])
            obstime_arr = astropy.time.Time(mjd_arr, format='mjd')
            midpoint_arr = obstime_arr + astropy.time.TimeDelta(exptime_arr / 2, format='sec')
            obstime_dt = obstime_arr.to_datetime()
            midpoint_dt = midpoint_arr.to_datetime()

            insert_rows = []
            for idx, row in enumerate(new_exposure_rows):
                night_num = int(row['night'])
                expnum = int(row['EXPNUM'])
                insert_rows.append({
                    "night_id": night_map[night_num],
                    "field_id": field_map.get(expnum, None),
                    "ra": float(ra_deg[idx]),
                    "dec": float(dec_deg[idx]),
                    "hp_index": int(hp_index[idx]),
                    "expnum": expnum,
                    "caldat": str(row['caldat']),
                    "exposure": float(row['exposure']),
                    "obstime": obstime_dt[idx],
                    "midpoint": midpoint_dt[idx],
                    "mjd": float(row['mjd']),
                    "midpoint_mjd": float(row['mjd_midpoint']),
                    "band": str(row['band']),
                    "target": str(row['OBJECT']),
                })

            for chunk in _chunked(insert_rows):
                session.execute(insert(Exposure), chunk)
            session.commit()

        expnum_to_exposure_id = dict(
            session.execute(select(Exposure.expnum, Exposure.id)).all()
        )

        # --- DetectorExposures: bulk insert, skipping existing (exposure, detector) pairs ---
        existing_pairs = set(
            session.execute(
                select(DetectorExposure.exposure_id, DetectorExposure.detector_id)
            ).all()
        )

        valid_detectors = [
            d for d in range(1, 63) if ccdBounds.get(ccd_num_to_name[d]) is not None
        ]

        # Build the list of (row, detector_number) combos that need inserting
        combos = []
        for row in exposures:
            expnum = int(row['EXPNUM'])
            exposure_id = expnum_to_exposure_id.get(expnum)
            if exposure_id is None:
                continue
            for detector in valid_detectors:
                detector_id = detector_map[detector]
                if (exposure_id, detector_id) in existing_pairs:
                    continue
                combos.append((row, detector, exposure_id, detector_id))

        if combos:
            n = len(combos)
            base_ra = np.empty(n)
            base_dec = np.empty(n)
            xmin_arr = np.empty(n)
            xmax_arr = np.empty(n)
            ymin_arr = np.empty(n)
            ymax_arr = np.empty(n)

            for idx, (row, detector, exposure_id, detector_id) in enumerate(combos):
                base_ra[idx] = float(row['RA(deg)'])
                base_dec[idx] = float(row['DEC(deg)'])
                xmin, xmax, ymin, ymax = ccdBounds[ccd_num_to_name[detector]]
                xmin_arr[idx] = xmin
                xmax_arr[idx] = xmax
                ymin_arr[idx] = ymin
                ymax_arr[idx] = ymax

            x_center = (xmax_arr + xmin_arr) / 2
            y_center = (ymax_arr + ymin_arr) / 2

            ra_c, dec_c, hp_c = ra_dec_to_coordinate_array(
                (base_ra + x_center) * u.deg, (base_dec + y_center) * u.deg
            )
            ra_00, dec_00, hp_00 = ra_dec_to_coordinate_array(
                (base_ra + xmin_arr) * u.deg, (base_dec + ymin_arr) * u.deg
            )
            ra_01, dec_01, hp_01 = ra_dec_to_coordinate_array(
                (base_ra + xmax_arr) * u.deg, (base_dec + ymin_arr) * u.deg
            )
            ra_10, dec_10, hp_10 = ra_dec_to_coordinate_array(
                (base_ra + xmin_arr) * u.deg, (base_dec + ymax_arr) * u.deg
            )
            ra_11, dec_11, hp_11 = ra_dec_to_coordinate_array(
                (base_ra + xmax_arr) * u.deg, (base_dec + ymax_arr) * u.deg
            )

            insert_rows = []
            for idx, (row, detector, exposure_id, detector_id) in enumerate(combos):
                insert_rows.append({
                    "detector_id": detector_id,
                    "exposure_id": exposure_id,
                    "ra": float(ra_c[idx]),
                    "dec": float(dec_c[idx]),
                    "hp_index": int(hp_c[idx]),
                    "ra_00": float(ra_00[idx]),
                    "dec_00": float(dec_00[idx]),
                    "hp_index_00": int(hp_00[idx]),
                    "ra_01": float(ra_01[idx]),
                    "dec_01": float(dec_01[idx]),
                    "hp_index_01": int(hp_01[idx]),
                    "ra_10": float(ra_10[idx]),
                    "dec_10": float(dec_10[idx]),
                    "hp_index_10": int(hp_10[idx]),
                    "ra_11": float(ra_11[idx]),
                    "dec_11": float(dec_11[idx]),
                    "hp_index_11": int(hp_11[idx]),
                })

            for chunk in _chunked(insert_rows):
                session.execute(insert(DetectorExposure), chunk)
            session.commit()


if __name__ == "__main__":
    main()
