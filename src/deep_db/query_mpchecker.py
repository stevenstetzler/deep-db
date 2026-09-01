import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astropy.table import vstack

from jorbit.mpchecker import mpchecker
from numpy import array
from sbident import SBIdent
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, selectinload, contains_eager
import logging
import sys
from sys import stderr
from collections import defaultdict
from joblib import Parallel, delayed

from shapely.geometry import box
from shapely.strtree import STRtree
from shapely.geometry import Point
from .models import DetectorExposure, Detector, Exposure

logger = logging.getLogger(__name__)

def get_objects(detector_exposure):
    radius = max(
        detector_exposure.ra_11 - detector_exposure.ra_00,
        detector_exposure.dec_11 - detector_exposure.dec_00
    ) / 2 * u.deg
    logger.info("checking Exposure=%s Detector=%s at (%f, %f) with radius %f degrees", detector_exposure.exposure.expnum, detector_exposure.detector.number, detector_exposure.ra, detector_exposure.dec, radius.value)
    mpc = mpchecker(
        SkyCoord(detector_exposure.ra * u.deg, detector_exposure.dec * u.deg),
        Time(detector_exposure.exposure.midpoint_mjd, format='mjd', scale='utc'),
        radius = radius
    )
    mpc = list(
        filter(
            lambda x : 
                (detector_exposure.ra_00 < x['ra']) and
                (x['ra'] < detector_exposure.ra_11) and
                (detector_exposure.dec_00 < x['dec']) and
                (x['dec'] < detector_exposure.dec_11),
            mpc
        )
    )
    return mpc

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query MPC ephemeris data for Deep DB")
    parser.add_argument("--db", required=True, help="Database URL")
    parser.add_argument("--processes", "-J", type=int, default=1, help="Number of processes to use")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements")
    parser.add_argument("--filter", type=str, nargs="+")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument(
        "--log-level", 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )

    engine = create_engine(args.db, echo=args.echo)
    with Session(engine) as session:
        q = (
            session.query(Exposure, Detector, DetectorExposure)
            .join(DetectorExposure, DetectorExposure.exposure_id == Exposure.id)
            .join(Detector, DetectorExposure.detector_id == Detector.id)
            .order_by(Exposure.id, Detector.id)
        )
        for filt in args.filter:
            k, v = filt.split("=")
            if k == "expnum":
                q = q.filter(Exposure.expnum == int(v))
            elif k == "number":
                q = q.filter(Detector.number == int(v))
            else:
                raise Exception(f"filtering on {k} not supported")

        grouped_exposures = defaultdict(list)

        for exposure, detector, detector_exposure in q:
            grouped_exposures[exposure].append((detector, detector_exposure))

        def get_info(exposures):
            for exposure, detectors in exposures:
                detector_exposures = list(map(lambda x : x[1], detectors))
                min_ra = min(map(lambda x : x.ra_00, detector_exposures))
                max_ra = max(map(lambda x : x.ra_11, detector_exposures))
                min_dec = min(map(lambda x : x.dec_00, detector_exposures))
                max_dec = max(map(lambda x : x.dec_11, detector_exposures))
                midpoint = SkyCoord((max_ra + min_ra)/2 * u.deg, (max_dec + min_dec)/2 * u.deg)
                radius = max(abs(max_ra - min_ra), abs(max_dec - min_dec)) / 2
                # print(midpoint, min_ra, max_ra, min_dec, max_dec)
                # midpoint = SkyCoord(exposure.ra * u.deg, exposure.dec * u.deg)
                # print(midpoint)
                yield exposure, midpoint, Time(exposure.midpoint_mjd, format='mjd', scale='utc'), radius * u.deg

        def get_objects(coord, time, radius):
            yield from mpchecker(
                coord,
                time,
                radius = radius,
                extra_precision=True,
                extra_precision_gravity="default solar system",
                observer="cerro tololo observatory, la serena",
            )

        did_header = False
        for exposure, coord, time, radius in get_info(grouped_exposures.items()):
            detectors = grouped_exposures[exposure]
            boxes = [
                box(
                    min(de.ra_00, de.ra_11), min(de.dec_00, de.dec_11),
                    max(de.ra_00, de.ra_11), max(de.dec_00, de.dec_11)
                )
                for _, de in detectors
            ]
            tree = STRtree(boxes)
            for obj in get_objects(coord, time, radius):
                if not args.no_header and not did_header:
                    print("|".join(["expnum", "detector"] + list(map(str, obj.columns))))
                    did_header = True
                point = Point(obj['ra'], obj['dec'])
                match_index = tree.query(point)
                if len(match_index) > 0:
                    matched_detector, matched_det_exp = detectors[match_index[0]]
                    print("|".join(map(str, [exposure.expnum, matched_detector.number] + list(obj))))
                    # print("|".join(map(str, [exposure.expnum, matched_detector.number, obj['Packed designation'], obj['Unpacked Name'], obj['ra'], obj['dec'], obj['est. Vmag'][0]])))

        # for exposure, coord, time, radius in get_info(q):
        #     for obj in get_objects(coord, time, radius):
        #         if obj['ra']
        #         print(exposure.expnum, obj['Unpacked Name'], obj['ra'], obj['dec'], obj['est. Vmag'][0])
        
        # get time and central pointing of exposures / min+max of ra/dec on
        # q = session.query(DetectorExposure).join(
        #     Exposure, Exposure.id == DetectorExposure.exposure_id
        # ).join(
        #     Detector, Detector.id == DetectorExposure.detector_id
        # )
        # q = q.options(
        #     contains_eager(DetectorExposure.exposure),
        #     contains_eager(DetectorExposure.detector),
        # )
        # if args.filter:
        #     for filt in args.filter:
        #         k, v = filt.split("=")
        #         if k == "number":
        #             q = q.filter(Detector.number == int(v))
        #         elif k == "expnum":
        #             q = q.filter(Exposure.expnum == int(v))
        #         else:
        #             raise Exception(f"filter on {k} not supported")
        # objects = Parallel(n_jobs=args.processes)(delayed(get_objects)(detector_exposure) for detector_exposure in q)
        # objects = sum(objects, [])
        # if len(objects) > 0:
        #     objects = vstack(objects)
        #     if args.no_header:
        #         objects.write(sys.stdout, format='ascii.no_header', delimiter=',')
        #     else:
        #         objects.write(sys.stdout, format='ascii.ecsv')
# mpc = mpchecker(
#     coordinate=c,
#     time=t,
#     radius=10 * u.arcmin,
#     extra_precision=True,
#     observer="cerro tololo observatory, la serena",
# )
# jpl = SBIdent(
#     "W84", t, c,
#     hwidth=r.to(u.deg).value, precision="high"
# ).results

# jpl_coords = SkyCoord(
#     ra=list(map(lambda x : Angle(x, unit=u.hourangle), jpl['Astrometric RA (hh:mm:ss)'])), 
#     dec=list(map(lambda x : Angle(x, unit=u.deg), jpl['Astrometric Dec (dd mm\'ss")'])), 
# )
# jpl = jpl_coords[
#     array(list(map(c.separation(jpl_coords))) < r)
# ]

# # print(mpc)
# print(jpl)


if __name__ == "__main__":
    main()
