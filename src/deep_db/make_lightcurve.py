from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from .models import Exposure, Ephemeris, DetectorExposure, EphemerisDetectorLocation, Photometry, SolarSystemObject, Night, Base
import lsst.afw.display as afwDisplay
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import astropy.table
import matplotlib
matplotlib.use('Agg')

def query_data(db, object_id, dataset):
    engine = create_engine(db)
    prev_night = None
    with Session(engine) as session:
        data = []
        for photometry, _, _, expnum, time, object_name, object_type, night in session.query(
            Photometry, DetectorExposure.id, Ephemeris.id,
            Exposure.expnum, Exposure.midpoint_mjd, 
            SolarSystemObject.name, SolarSystemObject.type,
            Night.night
        ).join(
            EphemerisDetectorLocation,
            Photometry.ephemeris_detector_location_id == EphemerisDetectorLocation.id
        ).join(
            Ephemeris,
            EphemerisDetectorLocation.ephemeris_id == Ephemeris.id
        ).join(
            DetectorExposure,
            Ephemeris.detector_exposure_id == DetectorExposure.id
        ).join(
            Exposure,
            DetectorExposure.exposure_id == Exposure.id
        ).join(
            SolarSystemObject,
            Ephemeris.object_id == SolarSystemObject.id,
        ).join(
            Night,
            Exposure.night_id == Night.id
        ).filter(
            Ephemeris.object_id == object_id
        ).filter(
            EphemerisDetectorLocation.dataset == dataset
        ).order_by(Exposure.expnum):
            if prev_night is None:
                prev_night = night
            if night != prev_night:
                if data:
                    yield astropy.table.Table(data)
                data = []
                prev_night = night
            d = {
                "name": object_name,
                "type": object_type,
                "expnum": expnum,
                "time": time,
                "mag": photometry.mag,
                "mag_err_lo": photometry.mag_err_lo, # TODO: fix this
                "mag_err_hi": photometry.mag_err_hi,
                "flux": photometry.flux_ref,
                "flux_err": photometry.sigma_ref,
                "mask": photometry.mask,
                "night": night
            }
            for k in ['mag', 'flux', 'flux_err', 'mag_err_lo', 'mag_err_hi']:
                if d[k] is None:
                    d[k] = np.nan
            data.append(d)
        if data:
            yield astropy.table.Table(data)

def light_curve(data, night=None):
    fig, ax = plt.subplots(facecolor='white')
    # print([data['mag_err_lo'], data['mag_err_hi']])
    ax.errorbar(
        data['expnum'],
        data['mag'],
        yerr=[data['mag_err_lo'], data['mag_err_hi']],
        fmt='o',
        capsize=2
    )
    ax.set_xlabel('EXPNUM')
    ax.set_ylabel('Mag')
    ax.tick_params()
    ax.invert_yaxis()
    if night:
        t = f"Light Curve for {data[0]['name']} on night {data[0]['night']}"
    else:
        t = f"Light Curve for {data[0]['name']}"
    ax.set_title(t)
    ax.grid(True)
    return fig, ax


def main():
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to process")
    parser.add_argument("--object-ids", type=int, nargs="+", default=[], help="List of object IDs to process")
    parser.add_argument("--object-names", type=str, nargs="+", default=[], help="List of object names to process")
    parser.add_argument("--object-types", type=str, nargs="+", default=[], help="List of object types to process")
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("./"))
    parser.add_argument("--echo", action="store_true")

    args = parser.parse_args()

    engine = create_engine(args.db, echo=args.echo)
    Base.metadata.create_all(engine)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def work(object_id):
        # print(astropy.table.Table(list(query_data(args.db, object_id, args.dataset))))
        try:
            all_data = []
            for data in query_data(args.db, object_id, args.dataset):
                all_data.append(data)
                night = data[0]['night']
                name = data[0]['name']
                object_type = data[0]['type']

                d = args.output_dir / str(night) / object_type / name
                d.mkdir(parents=True, exist_ok=True)
                
                fig, ax = light_curve(data)
                for ext in ['png']:#, 'jpg', 'pdf']:
                    f = d / f"light_curve_{args.dataset}.{ext}"
                    print(f"Saving light curve for object ID {object_id} on night {night} to {f}")
                    fig.savefig(f)
                plt.close(fig)
            
            if all_data:
                combined = astropy.table.vstack(all_data)
                # night = combined[0]['night']
                name = combined[0]['name']
                object_type = combined[0]['type']

                d = args.output_dir / "survey" / object_type / name
                d.mkdir(parents=True, exist_ok=True)
                fig, ax = light_curve(combined)
                for ext in ['png']:#, 'jpg', 'pdf']:
                    f = d / f"light_curve_{args.dataset}.{ext}"
                
                    print(f"Saving combined light curve for object ID {object_id} to {f}")
                    fig.savefig(f)
                plt.close(fig)
        except Exception as e:
            print(f"Error processing object ID {object_id}: {e}")

    def get_ids():
        with Session(engine) as session:
            q = session.query(SolarSystemObject.id).join(
                Ephemeris,
                SolarSystemObject.id == Ephemeris.object_id
            ).join(
                EphemerisDetectorLocation,
                Ephemeris.id == EphemerisDetectorLocation.ephemeris_id
            ).join(
                Photometry,
                EphemerisDetectorLocation.id == Photometry.ephemeris_detector_location_id
            ).filter(
                EphemerisDetectorLocation.dataset == args.dataset
            )
            if args.object_ids:
                q = q.filter(SolarSystemObject.id.in_(args.object_ids))

            if args.object_names:
                q = q.filter(SolarSystemObject.name.in_(args.object_names))
            
            if args.object_types:
                q = q.filter(SolarSystemObject.type.in_(args.object_types))

            q = q.distinct()
            for (obj_id,) in q:
                yield obj_id
    
    # print(list(get_ids()))
    Parallel(n_jobs=args.processes)(delayed(work)(obj_id) for obj_id in get_ids())

if __name__ == "__main__":
    main()
