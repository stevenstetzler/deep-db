from sqlalchemy.orm import Session
from sqlalchemy import func, String
from sqlalchemy import create_engine
from joblib import Parallel, delayed
import numpy as np
from .models import Base, Repository, Ephemeris, Detector, Exposure, DetectorExposure, EphemerisDetectorLocation, Cutout, Photometry, SolarSystemObject


def make_afw_image(image, variance, mask):
    import lsst.afw.image as afwImage
    exposure = afwImage.ExposureF(image.shape[1], image.shape[0])
    exposure.getMaskedImage().getImage().getArray()[:, :] = image
    exposure.getMaskedImage().getVariance().getArray()[:, :] = variance
    exposure.getMaskedImage().getMask().getArray()[:, :] = mask
    return exposure

def convert_zp(flux, orig, new):
    return flux * 10**(-2/5*(orig - new))

def logL_position(e, p, shift, ref_zp=31):
    import lsst.afw.image
    import scipy.ndimage
    eBBox = e.getBBox()
    psfBBox = e.psf.computeImageBBox(p)
    psfBBox.clip(eBBox)

    pc = e.getPhotoCalib()
    zp = pc.instFluxToMagnitude(1)

    mask = np.bitwise_or.reduce(lsst.afw.image.MaskX(e.mask, psfBBox).array.flatten()) # this is the bitwise or across all mask values for pixels that overlap the psf model
    model = lsst.afw.image.ImageD(e.psf.computeImage(p), psfBBox).array
    model = scipy.ndimage.shift(model, shift)
    stamp = lsst.afw.image.ImageD(lsst.afw.image.ImageF(e.image, psfBBox), deep=True).array
    weights = 1/lsst.afw.image.ImageD(lsst.afw.image.ImageF(e.variance, psfBBox), deep=True).array # inverse variance
    c = np.sum(model * stamp * weights, axis=(0, 1)) # Signal; Whidden et al. 2019 Eq. 19 (Psi)
    a = np.sum(model * model * weights, axis=(0, 1)) # Noise; Whidden et al. 2019 Eq. 20 (Phi)
    f = c/a # flux estimate; Whidden et al. 2019 Eq. 22 (alpha_ML)
    sigma = 1/np.sqrt(a) # standard deviation in flux estimate
    snr = c / np.sqrt(a) # signal to noise -- why is this not f / sigma?; Whidden et al. 2019 Eq. 26 (nu_coadd)
    mag = e.getPhotoCalib().instFluxToMagnitude(f)
    mag_lo = e.getPhotoCalib().instFluxToMagnitude(f - sigma) # this should be larger than mag
    mag_hi = e.getPhotoCalib().instFluxToMagnitude(f + sigma) # this should be smaller than mag
    mag_err_lo = mag - mag_hi
    mag_err_hi = mag_lo - mag

    logL = -0.5 * np.sum(weights * (f * model - stamp) ** 2)
    return {
        "logL": logL, 
        "a": a,
        "c": c,
        "flux": f, 
        "flux_ref": convert_zp(f, zp, ref_zp),
        "sigma": sigma, 
        "sigma_ref": convert_zp(sigma, zp, ref_zp),
        "SNR": snr,
        "mag": mag,
        "mag_err_lo": mag_err_lo,
        "mag_err_hi": mag_err_hi,
        "mask": mask,
        "zero_point": zp,
    }

def map_types(value):
    import numpy as np
    import pandas as pd
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    elif isinstance(value, np.bool_): # Add an explicit check for NumPy boolean
        return bool(value)
    elif isinstance(value, (np.datetime64, pd.Timestamp)):
        # Convert to an ISO-formatted string
        return pd.to_datetime(value).isoformat()    
    elif isinstance(value, np.ndarray):
        return list(map(map_types, value))
    return value

# def forced_exposures(exposures, points):
#     import lsst.geom
#     import astropy.table
    
#     results = []
#     for exposure, (x, y) in zip(exposures, points):
#         result = logL_position(exposure, lsst.geom.Point2D(x, y), [0, 0])
#         result = {
#             f"forced_{key}": value
#             for key, value in result.items()
#         }
#         result['forced_i_x'] = x
#         result['forced_i_y'] = y
#         result['forced_exposure'] = exposure.getInfo().getVisitInfo().getId()
#         result['forced_detector'] = exposure.getDetector().getId()
#         result['forced_time'] = exposure.getInfo().getVisitInfo().date.toAstropy()
#         results.append(result)

#     return astropy.table.Table(results)


def main():
    import argparse 
    from pathlib import Path
    import socket
    import lsst.daf.butler as dafButler
    import lsst.geom
    import sys
    import json

    parser = argparse.ArgumentParser(description="Ingest location data into Deep DB")
    parser.add_argument("repo", type=str, help="Path to the repository")
    parser.add_argument("dataset", type=str, help="Name of the dataset to query")
    parser.add_argument("--collections", type=str)
    parser.add_argument("--db", type=str)
    parser.add_argument("--repo-name", type=str)
    parser.add_argument("--cutout-size", type=int, default=100, help="Size of the cutout in pixels (default: 100)")
    parser.add_argument("--processes", type=int, default=1, help="Number of parallel processes to use (default: 1)")
    parser.add_argument("--object-ids", type=int, nargs="+", default=[], help="List of object IDs to process")
    parser.add_argument("--object-names", type=str, nargs="+", default=[], help="List of object names to process")
    parser.add_argument("--object-types", type=str, nargs="+", default=[], help="List of object types to process")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements")

    args = parser.parse_args()
    _butler = dafButler.Butler(args.repo)
    butler = dafButler.Butler(args.repo, collections=_butler.registry.queryCollections(args.collections))

    repo_path = Path(args.repo).resolve()
    repo_name = args.repo_name if args.repo_name else repo_path.name

    engine = create_engine(args.db, echo=args.echo)
    Base.metadata.create_all(engine)

    def query_location(dataset, repository_id, detector_exposure_id, ephemeris_ids):
        print("Processing DetectorExposure id", detector_exposure_id, file=sys.stderr)
        # group by detector/exposure
        # get exposure.wcs to get locations
        # load exposure
        # get and put cutouts from exposure
        engine = create_engine(args.db, echo=args.echo)
        with Session(engine) as session:
            result = session.query(
                DetectorExposure.id, Exposure.expnum, Detector.number
            ).join(
                Detector, DetectorExposure.detector_id == Detector.id
            ).join(
                Exposure, DetectorExposure.exposure_id == Exposure.id
            ).filter(
                DetectorExposure.id == detector_exposure_id
            ).first()
            if result is None:
                raise ValueError(f"DetectorExposure with id {detector_exposure_id} not found")
            _, visit, detector = result
            
            try:
                ref = next(iter(butler.registry.queryDatasets(dataset, where=f"instrument='DECam' and visit={visit} and detector={detector}")))
            except StopIteration:
                print(f"No {dataset} found for visit {visit} and detector {detector}, skipping", file=sys.stderr)
                return

            collection = ref.run
            exposure = butler.get(ref)
            wcs = exposure.getWcs()
            bbox = exposure.getBBox()
            for ephemeris_id in ephemeris_ids:
                ephemeris = session.query(Ephemeris).filter_by(id=ephemeris_id).first()
                if ephemeris is None:
                    print(f"Ephemeris with id {ephemeris_id} not found, skipping", file=sys.stderr)
                    continue          

                ra = ephemeris.ra
                dec = ephemeris.dec
                sp = lsst.geom.SpherePoint(
                    lsst.geom.Angle(ra, lsst.geom.degrees), 
                    lsst.geom.Angle(dec, lsst.geom.degrees)
                )

                ephemeris_detector_location = session.query(EphemerisDetectorLocation).filter_by(
                    ephemeris_id=ephemeris_id,
                    repository_id=repository_id,
                    collection=collection,
                    dataset=dataset,
                ).first()
                if ephemeris_detector_location is not None:
                    print(f"EphemerisDetectorLocation for ephemeris_id {ephemeris_id}, repository_id {repository_id}, collection {collection}, dataset {dataset} already exists", file=sys.stderr)
                    x, y = ephemeris_detector_location.x, ephemeris_detector_location.y
                    p = lsst.geom.Point2D(x, y)
                else:
                    p = wcs.skyToPixel(sp)
                    x, y = p.getX(), p.getY()
                    try:
                        if not bbox.contains(lsst.geom.Point2I(int(x), int(y))):
                            print(f"Ephemeris id {ephemeris_id} with RA {ra}, Dec {dec} is outside the detector bounds, skipping", file=sys.stderr)
                            continue
                    except Exception as e:
                        print(f"Error checking bounds for Ephemeris id {ephemeris_id} with RA {ra}, Dec {dec}: {e}, skipping", file=sys.stderr)
                        continue
                    ephemeris_detector_location = EphemerisDetectorLocation(
                        ephemeris_id=ephemeris.id,
                        repository_id=repository_id,
                        collection=collection,
                        dataset=dataset,
                        x=float(x),
                        y=float(y)
                    )
                    session.add(ephemeris_detector_location)

                cutout = session.query(Cutout).filter_by(
                    ephemeris_detector_location=ephemeris_detector_location
                ).first()
                if cutout is not None:
                    print(f"Cutout for EphemerisDetectorLocation id {ephemeris_detector_location.id} already exists", file=sys.stderr)
                else:
                    cutout_bbox = lsst.geom.Box2I(
                        lsst.geom.Point2I(
                            int(x - args.cutout_size // 2), 
                            int(y - args.cutout_size // 2)
                        ), 
                        lsst.geom.Extent2I(args.cutout_size, args.cutout_size)
                    )
                    cutout_bbox.clip(bbox)
                    cutout_data = exposure.getCutout(cutout_bbox)

                    cutout = Cutout(
                        image=[(None if np.isnan(x) else float(x)) for x in cutout_data.image.array.flatten()],
                        variance=[(None if np.isnan(x) else float(x)) for x in cutout_data.variance.array.flatten()],
                        mask=[int(x) for x in cutout_data.mask.array.flatten()],
                        width=cutout_data.getBBox().getWidth(),
                        height=cutout_data.getBBox().getHeight(),
                        ephemeris_detector_location=ephemeris_detector_location,
                    )
                    session.add(cutout)

                photometry = session.query(Photometry).filter_by(
                    ephemeris_detector_location=ephemeris_detector_location
                ).first()
                if photometry is not None:
                    print(f"Photometry for EphemerisDetectorLocation id {ephemeris_detector_location.id} already exists", file=sys.stderr)
                else:
                    photometry_data = logL_position(exposure, p, [0, 0])
                    photometry_data = {k: map_types(v) for k, v in photometry_data.items()}
                    photometry = Photometry(
                        logl=photometry_data['logL'],
                        a=photometry_data['a'],
                        c=photometry_data['c'],
                        flux=photometry_data['flux'],
                        flux_ref=photometry_data['flux_ref'],
                        sigma=photometry_data['sigma'],
                        sigma_ref=photometry_data['sigma_ref'],
                        snr=photometry_data['SNR'],
                        mag=photometry_data['mag'],
                        mag_err_lo=photometry_data['mag_err_lo'],
                        mag_err_hi=photometry_data['mag_err_hi'],
                        mask=photometry_data['mask'],
                        zero_point=photometry_data['zero_point'],
                        ephemeris_detector_location=ephemeris_detector_location,
                    )
                    session.add(photometry)

            session.commit()

    with Session(engine) as session:
        repo = session.query(Repository).filter_by(
            name=repo_name,
            host=socket.gethostname(),
            path=str(repo_path)
        ).first()
        if not repo:
            repo = Repository(
                name=repo_name,
                host=socket.gethostname(),
                path=str(repo_path)
            )
            session.add(repo)
            session.commit()
        
        q = session.query(
            Ephemeris.detector_exposure_id, func.string_agg(Ephemeris.id.cast(String), ',')
        ).group_by(Ephemeris.detector_exposure_id)

        if args.object_ids:
            q = q.filter(Ephemeris.object_id.in_(args.object_ids))
        if args.object_names or args.object_types:
            q = q.join(SolarSystemObject, Ephemeris.object_id == SolarSystemObject.id)
        if args.object_names:
            q = q.filter(SolarSystemObject.name.in_(args.object_names))
        if args.object_types:
            q = q.filter(SolarSystemObject.type.in_(args.object_types))

        # print(list(q))
        Parallel(n_jobs=args.processes)(
            delayed(query_location)(args.dataset, repo.id, detector_exposure_id, list(map(int, ephemeris_ids.split(','))))
            for detector_exposure_id, ephemeris_ids in q
        )
        # for detector_exposure_id, ephemeris_ids in q:
        #     print(f"Processing DetectorExposure id {detector_exposure_id} with Ephemeris ids {ephemeris_ids}", file=sys.stderr)
        #     query_location(args.dataset, repo.id, detector_exposure_id, list(map(int, ephemeris_ids.split(','))))
        #     break
        
if __name__ == "__main__":
    main()
