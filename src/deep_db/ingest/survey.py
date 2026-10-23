def ingest_survey(exposures, field_table):
    from astropy.time import Time, TimeDelta
    from astropy.units import deg
    from sqlalchemy.orm import Session
    from deep_db.database.survey import Detector, Field, Night, Exposure, DetectorExposure
    from deep_db.database import get_db
    from deep_db.util.decam import ccdBounds, ccd_num_to_name
    from deep_db.util.coord import ra_dec_to_coordinate

    engine = get_db()
    with Session(engine) as session:
        existing_dets = {d.number: d.id for d in session.query(Detector).all()}
        missing_dets = [
            Detector(number=int(det_num), name=ccd_num_to_name[det_num])
            for det_num in range(1, 63)
            if det_num not in existing_dets
        ]
        if missing_dets:
            session.add_all(missing_dets)
            session.flush()
            # Update map with newly inserted IDs
            existing_dets.update({d.number: d.id for d in missing_dets})
        detector_map = existing_dets
        
        existing_fields = {f.name: f.id for f in session.query(Field.name, Field.id).all()}
        new_fields = {}
        for row in field_table:
            field_name = str(row['field'])
            expnum = row['expnum']
            if field_name not in existing_fields and field_name not in new_fields:
                new_fields[field_name] = Field(name=field_name)

        if new_fields:
            session.add_all(new_fields.values())
            session.flush()
            existing_fields.update({name: obj.id for name, obj in new_fields.items()})

        field_map = {row['expnum']: existing_fields[str(row['field'])] for row in field_table}

        unique_nights = {int(row['night']) for row in exposures}
        start_stop = {
            rows[0]['night']: (
                min(
                    Time(rows['mjd'], format='mjd', scale='utc')
                ).datetime,
                max(
                    Time(rows['mjd'], format='mjd', scale='utc') + TimeDelta(rows['exposure'], format='sec')
                ).datetime
            )
            for rows in exposures.group_by("night").groups
        }
        existing_nights = {
            n.night: n.id 
            for n in session.query(Night.night, Night.id).filter(Night.night.in_(unique_nights)).all()
        }
        
        missing_nights = [
            Night(
                night=int(night_num), 
                start_time=start_stop[night_num][0], 
                end_time=start_stop[night_num][1]
            )
            for night_num in unique_nights
            if night_num not in existing_nights
        ]
        if missing_nights:
            session.add_all(missing_nights)
            session.flush()
            existing_nights.update({n.night: n.id for n in missing_nights})
        night_map = existing_nights

        existing_expnums = set(
            e.expnum for e in session.query(Exposure.expnum).all()
        )
        det_exposures_to_add = []

        for row in exposures:
            expnum = int(row['EXPNUM'])
            if expnum in existing_expnums:
                continue
            existing_expnums.add(expnum)

            night_num = int(row['night'])
            obstime = Time(row['mjd'], format='mjd', scale='utc')
            midpoint = obstime + TimeDelta(row['exposure'] / 2, format='sec')
            ra, dec, hp_index = ra_dec_to_coordinate(row['RA(deg)'] * deg, row['DEC(deg)'] * deg)

            exposure = Exposure(
                night_id=int(night_map[night_num]),
                field_id=field_map.get(expnum, None),
                ra=float(ra),
                dec=float(dec),
                hp_index=int(hp_index),
                expnum=int(expnum),
                caldat=str(row['caldat']),
                exposure_time=float(row['exposure']),
                obstime=obstime.utc.datetime,
                midpoint=midpoint.utc.datetime,
                band=str(row['band']),
                target=str(row['OBJECT']),
            )
            session.add(exposure)
            session.flush()  # Flush once per exposure to generate exposure.id for children

            row_ra = row['RA(deg)']
            row_dec = row['DEC(deg)']

            # Generate all 62 DetectorExposures in memory
            for detector in range(1, 63):
                ccd_name = ccd_num_to_name[detector]
                bounds = ccdBounds.get(ccd_name, None)
                if bounds is None:
                    continue

                xmin, xmax, ymin, ymax = bounds
                
                ra_center, dec_center, hp_index_c = ra_dec_to_coordinate((row_ra + (xmax + xmin) / 2) * deg, (row_dec + (ymax + ymin) / 2) * deg)
                ra_00, dec_00, hp_index_00 = ra_dec_to_coordinate((row_ra + xmin) * deg, (row_dec + ymin) * deg)
                ra_01, dec_01, hp_index_01 = ra_dec_to_coordinate((row_ra + xmax) * deg, (row_dec + ymin) * deg)
                ra_10, dec_10, hp_index_10 = ra_dec_to_coordinate((row_ra + xmin) * deg, (row_dec + ymax) * deg)
                ra_11, dec_11, hp_index_11 = ra_dec_to_coordinate((row_ra + xmax) * deg, (row_dec + ymax) * deg)

                det_exposures_to_add.append(
                    DetectorExposure(
                        detector_id=int(detector_map[detector]),
                        exposure_id=int(exposure.id),
                        ra=float(ra_center),
                        dec=float(dec_center),
                        hp_index=int(hp_index_c),
                        ra_00=float(ra_00), dec_00=float(dec_00), hp_index_00=int(hp_index_00),
                        ra_01=float(ra_01), dec_01=float(dec_01), hp_index_01=int(hp_index_01),
                        ra_10=float(ra_10), dec_10=float(dec_10), hp_index_10=int(hp_index_10),
                        ra_11=float(ra_11), dec_11=float(dec_11), hp_index_11=int(hp_index_11),
                    )
                )

        # Bulk insert all accumulated DetectorExposures at once
        if det_exposures_to_add:
            session.add_all(det_exposures_to_add)

        session.commit()