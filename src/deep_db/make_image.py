from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from .models import Exposure, Ephemeris, DetectorExposure, EphemerisDetectorLocation, Cutout, SolarSystemObject, Night
import lsst.afw.display as afwDisplay
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
from joblib import Parallel, delayed

afwDisplay.setDefaultBackend('matplotlib')

def query_data(db, object_id, width, height, dataset, by_night=True):
    print("Querying data for object ID:", object_id)
    engine = create_engine(db)
    with Session(engine) as session:
        data = {
            "name": None,
            "type": None,
            "night": None,
            "expnums": [],
            "times": [],
            "image": [],
            "variance": [],
            "mask": []
        }
        prev_night = None
        for cutout, _, _, expnum, time, object_name, object_type, night in session.query(
            Cutout, DetectorExposure.id, Ephemeris.id,
            Exposure.expnum, Exposure.mjd, 
            SolarSystemObject.name, SolarSystemObject.type,
            Night.night
        ).join(
            EphemerisDetectorLocation,
            Cutout.ephemeris_detector_location_id == EphemerisDetectorLocation.id
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
            Cutout.width == width, Cutout.height == height
        ).filter(
            EphemerisDetectorLocation.dataset == dataset
        ).order_by(Exposure.expnum):
            if by_night:
                if prev_night is None:
                    prev_night = night
                if night != prev_night:
                    if len(data['expnums']) > 0:
                        data['image'] = np.array(data['image']).reshape((-1, width, height))
                        data['variance'] = np.array(data['variance']).reshape((-1, width, height))
                        data['mask'] = np.array(data['mask']).reshape((-1, width, height))
                        yield data
                    data = {
                        "name": None,
                        "type": None,
                        "night": None,
                        "expnums": [],
                        "times": [],
                        "image": [],
                        "variance": [],
                        "mask": []
                    }
                    prev_night = night

            # print(prev_night, night, expnum)
            data["name"] = object_name
            data["type"] = object_type
            data["night"] = night if by_night else "survey"
            data["expnums"].append(expnum)
            data["times"].append(time)
            data["image"].append(
                [(x if x is not None else float("NaN")) for x in cutout.image]
            )
            data["variance"].append(
                [(x if x is not None else float("NaN")) for x in cutout.variance]
            )
            data["mask"].append(cutout.mask)

        if len(data['expnums']) > 0:
            data['image'] = np.array(data['image']).reshape((-1, width, height))
            data['variance'] = np.array(data['variance']).reshape((-1, width, height))
            data['mask'] = np.array(data['mask']).reshape((-1, width, height))
            yield data

    #     name = object_name
    #     image_data = cutout.image #list(map(float, cutout.image.strip('[]').split(',')))
    #     variance_data = cutout.variance # list(map(float, cutout.variance.strip('[]').split(',')))
    #     mask_data = cutout.mask # list(map(int, cutout.mask.strip('[]').split(',')))
    #     expnums.append(expnum)
    #     times.append(time)
    #     images.append([(x if x is not None else float("NaN")) for x in image_data])
    #     variance.append([(x if x is not None else float("NaN")) for x in variance_data])
    #     mask.append(mask_data)

    # return {
    #     "name": name,
    #     "expnums": expnums,
    #     "times": times,
    #     "image": np.array(images).reshape((-1, width, height)),
    #     "variance": np.array(variance).reshape((-1, width, height)),
    #     "mask": np.array(mask).reshape((-1, width, height))
    # }


# Visualization functions

def scale_images(images, scale=ZScaleInterval, stretch=AsinhStretch):
    norm = ImageNormalize(
        images.flatten(),
        interval=scale(),
        stretch=stretch()
    )
    scaled_images = []
    for image in images:
        scaled_images.append(norm(image) * 255)
    return np.array(scaled_images)

def make_gif(images, duration=500, output_filename="output.gif"):
    images[0].save(
        output_filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved GIF as {output_filename}")

def stack(images, method="mean", variance=None):
    if method == "mean":
        return np.mean(images, axis=0)
    elif method == "median":
        return np.median(images, axis=0)
    elif method == "sum":
        return np.sum(images, axis=0)
    elif method == "weighted":
        if variance is None:
            raise ValueError("Variance must be provided for weighted stacking")
        weights = 1 / variance
        # TODO: check Pedro's code
        return np.sum(images * weights, axis=0) / np.sum(weights, axis=0)
    elif method == "or":
        return np.bitwise_or.reduce(images.astype(np.uint8), axis=0)
    elif method == "and":
        return np.bitwise_and.reduce(images.astype(np.uint8), axis=0)
    else:
        raise ValueError(f"Unknown stacking method: {method}")

def main():
    import argparse
    from PIL import Image
    import numpy as np
    from astropy.visualization import ZScaleInterval, ImageNormalize, simple_norm, AsinhStretch
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("image_type", type=str, help="One of: gif, stack")
    parser.add_argument("--db", type=str)
    parser.add_argument("--dataset", type=str, default='calexp')
    parser.add_argument("--layer", type=str, default="image")
    parser.add_argument("--stack-method", type=str, default="weighted")
    parser.add_argument("--object-ids", type=int, nargs="+", default=[])
    parser.add_argument("--object-names", type=str, nargs="+", default=[])
    parser.add_argument("--object-types", type=str, nargs="+", default=[])
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--by-night", action="store_true")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("./"))
    parser.add_argument("--echo", action="store_true")

    args = parser.parse_args()

    engine = create_engine(args.db, echo=args.echo)

    def get_ids():
        with Session(engine) as session:
            q = session.query(SolarSystemObject.id).join(
                Ephemeris,
                SolarSystemObject.id == Ephemeris.object_id
            ).join(
                EphemerisDetectorLocation,
                Ephemeris.id == EphemerisDetectorLocation.ephemeris_id
            ).join(
                Cutout,
                EphemerisDetectorLocation.id == Cutout.ephemeris_detector_location_id
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

    def work(object_id):
        try:
            all_data = []
            for data in query_data(
                args.db,
                object_id,
                args.width,
                args.height,
                args.dataset
            ):
                # print(len(data['expnums']), data['night'])
                # continue
                all_data.append(data)
                night = data['night']
                name = data['name']
                object_type = data['type']

                d = args.output_dir / str(night) / object_type / name
                d.mkdir(parents=True, exist_ok=True)
                if len(data.get(args.layer)) == 0:
                    print(f"Layer {args.layer} not found.")
                    continue

                if args.image_type == "stack":
                    stacked_image = stack(
                        data[args.layer],
                        method=args.stack_method,
                        variance=data.get("variance")
                    )
                    img = Image.fromarray(
                        scale_images(np.array([stacked_image]))[0]
                    ).convert("L")
                    f = d / f"stack_{args.layer}_{args.dataset}_{args.stack_method}.png"
                    img.save(f)
                    print(f"Saved stacked image as {f}")
                elif args.image_type == "gif":
                    images = [
                        Image.fromarray(
                            img.astype(np.float32)
                        )
                        for i, img in enumerate(scale_images(data[args.layer]))
                    ]
                    f = d / f"layer_{args.layer}_{args.dataset}.gif"
                    make_gif(images, output_filename=f)
                else:
                    raise ValueError(f"Unknown image type: {args.image_type}")
            
            if all_data:
                combined = {
                    "name": all_data[0]['name'],
                    "type": all_data[0]['type'],
                    "night": "survey",
                    "expnums": [],
                    "times": [],
                    "image": [],
                    "variance": [],
                    "mask": []
                }
                for data in all_data:
                    combined['expnums'].extend(data['expnums'])
                    combined['times'].extend(data['times'])
                    combined['image'].extend(data['image'])
                    combined['variance'].extend(data['variance'])
                    combined['mask'].extend(data['mask'])
                combined['image'] = np.array(combined['image']).reshape((-1, args.width, args.height))
                combined['variance'] = np.array(combined['variance']).reshape((-1, args.width, args.height))
                combined['mask'] = np.array(combined['mask']).reshape((-1, args.width, args.height))

                name = combined['name']
                object_type = combined['type']

                d = args.output_dir / "survey" / object_type / name
                d.mkdir(parents=True, exist_ok=True)
                if len(combined.get(args.layer)) == 0:
                    print(f"Layer {args.layer} not found.")
                    return

                if args.image_type == "stack":
                    stacked_image = stack(
                        combined[args.layer],
                        method=args.stack_method,
                        variance=combined.get("variance")
                    )
                    img = Image.fromarray(
                        scale_images(np.array([stacked_image]))[0]
                    ).convert("L")
                    f = d / f"stack_{args.layer}_{args.dataset}_{args.stack_method}.png"
                    img.save(f)
                    print(f"Saved stacked image as {f}")
                elif args.image_type == "gif":
                    images = [
                        Image.fromarray(
                            img.astype(np.float32)
                        )
                        for i, img in enumerate(scale_images(combined[args.layer]))
                    ]
                    f = d / f"layer_{args.layer}_{args.dataset}.gif"
                    make_gif(images, output_filename=f)
                else:
                    raise ValueError(f"Unknown image type: {args.image_type}")
        except Exception as e:
            print(f"Error processing object ID {object_id}: {e}")

        # if all_data:
        #     combined = astropy.table.vstack(all_data)
        #     # night = combined[0]['night']
        #     name = combined[0]['name']
        #     object_type = combined[0]['type']

        #     d = args.output_dir / "survey" / object_type / name
        #     d.mkdir(parents=True, exist_ok=True)
        #     fig, ax = light_curve(combined)
        #     for ext in ['png', 'jpg', 'pdf']:
        #         f = d / f"light_curve_{args.dataset}.{ext}"
            
        #         print(f"Saving combined light curve for object ID {object_id} to {f}")
        #         fig.savefig(f)
        #     plt.close(fig)


    # print(list(get_ids()))
    Parallel(n_jobs=args.processes)(delayed(work)(obj_id) for obj_id in get_ids())

    # with Session(engine) as session:
    #     for data in query_data(
    #         session,
    #         args.object_id,
    #         args.width,
    #         args.height,
    #         args.dataset,
    #         by_night=args.by_night
    #     ):
    #         if len(data.get(args.layer)) == 0:
    #             print(f"Layer {args.layer} not found.")
    #             return

    #         if args.image_type == "stack":
    #             stacked_image = stack(
    #                 data[args.layer],
    #                 method=args.stack_method,
    #                 variance=data.get("variance")
    #             )
    #             img = Image.fromarray(
    #                 scale_images(np.array([stacked_image]))[0]
    #             ).convert("L")
    #             img.save(f"object_{args.object_id}_stack_{args.stack_method}.png")
    #             print(f"Saved stacked image as object_{args.object_id}_stack_{args.stack_method}_layer_{args.layer}_dataset_{args.dataset}.png")
    #             return
    #         elif args.image_type == "gif":
    #             images = [
    #                 Image.fromarray(
    #                     img.astype(np.float32)
    #                 )
    #                 for i, img in enumerate(scale_images(data[args.layer]))
    #             ]
    #             make_gif(images, output_filename=f"object_{args.object_id}_layer_{args.layer}_dataset_{args.dataset}.gif")
    #         else:
    #             raise ValueError(f"Unknown image type: {args.image_type}")


if __name__ == "__main__":
    main()