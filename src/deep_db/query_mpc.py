from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func
import astropy.units as u
from datetime import timedelta
from joblib import Parallel, delayed
from time import sleep
from random import random
import os
from .models import Base, SolarSystemObject, Exposure, Night, DetectorExposure, ra_dec_to_coordinate, MPCTracklet, MPCObservation

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query MPC ephemeris data for Deep DB")
    parser.add_argument("--mpc", required=True, help="MPC ephemeris data URL")
    parser.add_argument("--db", required=True, help="Database URL")
    parser.add_argument("--processes", "-J", type=int, default=1, help="Number of processes to use")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements")

    args = parser.parse_args()

    deep_engine = create_engine(args.db, echo=args.echo)
    Base.metadata.create_all(deep_engine)

    mpc_engine = create_engine(args.mpc, echo=args.echo)
    metadata = MetaData()
    metadata.reflect(bind=mpc_engine)
    obs_sbn = Table("obs_sbn", metadata, autoload_with=mpc_engine)

    mpc_columns = ["trkid", "trksub", "permid", "provid", "ra", "dec", "mag", "band", "obstime", "status"]
    select = list(map(lambda x : getattr(obs_sbn.columns, x), mpc_columns))

    def add_objects(tmin, tmax, exposure):
        current_pid = os.getpid()
        deep_engine = create_engine(args.db)
        mpc_engine = create_engine(args.mpc)
        with Session(deep_engine) as deep_db, Session(mpc_engine) as mpc:
            mpc_query = mpc.query(
                obs_sbn.columns.provid
            ).filter(
                obs_sbn.columns.stn == "W84"
            ).filter(
                obs_sbn.columns.obstime > tmin
            ).filter(
                obs_sbn.columns.obstime < tmax + timedelta(seconds=exposure)
            ).group_by(obs_sbn.columns.provid)
            for (provid,) in mpc_query:
                obj = deep_db.query(SolarSystemObject).filter_by(name=provid, type='mpc').first()
                if obj is None:
                    obj = SolarSystemObject(name=provid, type='mpc')
                    deep_db.add(obj)
            deep_db.commit()

    def add_tracklets(tmin, tmax, exposure):
        current_pid = os.getpid()
        deep_engine = create_engine(args.db)
        mpc_engine = create_engine(args.mpc)
        with Session(deep_engine) as deep_db, Session(mpc_engine) as mpc:
            mpc_query = mpc.query(
                obs_sbn.columns.trksub, obs_sbn.columns.trkid
            ).filter(
                obs_sbn.columns.stn == "W84"
            ).filter(
                obs_sbn.columns.obstime > tmin
            ).filter(
                obs_sbn.columns.obstime < tmax + timedelta(seconds=exposure)
            ).group_by(obs_sbn.columns.trksub, obs_sbn.columns.trkid)
            for (trksub, trkid) in mpc_query:
                (provid,) = mpc.query(
                    obs_sbn.columns.provid
                ).filter(
                    obs_sbn.columns.stn == "W84",
                    obs_sbn.columns.trksub == trksub,
                    obs_sbn.columns.trkid == trkid,
                ).first()
                if provid is not None:
                    obj = deep_db.query(SolarSystemObject).filter_by(name=provid, type='mpc').first()
                    obj_id = obj.id
                else:
                    obj = None
                    obj_id = None
                
                mpc_tracklet = deep_db.query(MPCTracklet).filter_by(
                    object_id=obj_id,
                    trksub=trksub,
                    trkid=trkid,
                ).first()
                if mpc_tracklet is None:
                    mpc_tracklet = MPCTracklet(
                        object=obj,
                        trksub=trksub,
                        trkid=trkid,
                    )
                    deep_db.add(mpc_tracklet)
                    deep_db.commit()

    def add_observations(tmin, tmax, exposure):
        current_pid = os.getpid()
        num_missed = 0
        num_total = 0

        deep_engine = create_engine(args.db)
        mpc_engine = create_engine(args.mpc)
        with Session(deep_engine) as deep_db, Session(mpc_engine) as mpc:
            mpc_query = mpc.query(
                *select
            ).filter(
                obs_sbn.columns.stn == "W84"
            ).filter(
                obs_sbn.columns.obstime > tmin
            ).filter(
                obs_sbn.columns.obstime < tmax + timedelta(seconds=exposure)
            )
            for row in mpc_query:
                num_total += 1
                row = {c: v for c, v in zip(mpc_columns, row)}

                ra, dec, hp_index = ra_dec_to_coordinate(row['ra'] * u.deg, row['dec'] * u.deg)

                if row['provid'] is None:
                    obj_id = None
                else:
                    obj = deep_db.query(SolarSystemObject).filter(
                        SolarSystemObject.name==row['provid'],
                        SolarSystemObject.type=='mpc'
                    ).first()
                    obj_id = obj.id
                    if obj is None:
                        print(f"[{current_pid}] Missing SolarSystemObject for provid {row['provid']}")
                        num_missed += 1
                        continue
                
                mpc_tracklet = deep_db.query(MPCTracklet).filter(
                    MPCTracklet.object_id == obj_id,
                    MPCTracklet.trksub == row['trksub'],
                    MPCTracklet.trkid == row['trkid'],
                ).first()
                if mpc_tracklet is None:
                    print(f"[{current_pid}] Missing MPCTracklet for trksub {row['trksub']} trkid {row['trkid']} provid {row['provid']}")
                    num_missed += 1
                    continue

                de = deep_db.query(DetectorExposure).join(
                    Exposure, DetectorExposure.exposure_id == Exposure.id
                ).join(
                    Night, Exposure.night_id == Night.id
                ).filter(
                    row['obstime'] >= Exposure.obstime
                ).filter(
                    row['obstime'] <= Exposure.obstime + func.make_interval(0, 0, 0, 0, 0, 0, Exposure.exposure)
                ).filter(
                    DetectorExposure.ra_00 < ra,
                    DetectorExposure.ra_11 > ra,
                    DetectorExposure.dec_00 < dec,
                    DetectorExposure.dec_11 > dec,
                ).first()
                if de is None:
                    print(f"[{current_pid}] Missing DetectorExposure for hp_index {hp_index} ra {row['ra']} dec {row['dec']} trkid {row['trkid']} obstime {row['obstime']}")
                    num_missed += 1
                    continue

                obs = deep_db.query(MPCObservation).filter(
                    MPCObservation.tracklet_id==mpc_tracklet.id,
                    MPCObservation.detector_exposure_id==de.id,
                ).first()

                if obs is None:
                    obs = MPCObservation(
                        tracklet=mpc_tracklet,
                        detector_exposure=de,
                        ra=ra,
                        dec=dec,
                        obstime=row['obstime'],
                        hp_index=hp_index,
                        band=row['band'],
                        mag=row['mag'],
                        status=row['status'],
                    )
                    deep_db.add(obs)
                deep_db.commit()
        return num_missed, num_total

    num_missed = 0
    num_total = 0
    with Session(deep_engine) as deep_db:
        # night_query = deep_db.query(
        #     Night.night,
        #     func.min(Exposure.obstime),
        #     func.max(Exposure.obstime),
        #     func.max(Exposure.exposure)
        # ).join(
        #     Exposure, Exposure.night_id == Night.id
        # ).group_by(
        #     Night.night
        # ).order_by(
        #     Night.night
        # )
        # for night, tmin, tmax, exposure in night_query:
        #     add_objects(tmin, tmax, exposure)
        
        # night_query = deep_db.query(
        #     Night.night,
        #     func.min(Exposure.obstime),
        #     func.max(Exposure.obstime),
        #     func.max(Exposure.exposure)
        # ).join(
        #     Exposure, Exposure.night_id == Night.id
        # ).group_by(
        #     Night.night
        # ).order_by(
        #     Night.night
        # )
        # Parallel(n_jobs=args.processes)(delayed(add_tracklets)(tmin, tmax, exposure) for night, tmin, tmax, exposure in night_query)
        
        night_query = deep_db.query(
            Night.night,
            func.min(Exposure.obstime),
            func.max(Exposure.obstime),
            func.max(Exposure.exposure)
        ).join(
            Exposure, Exposure.night_id == Night.id
        ).group_by(
            Night.night
        ).order_by(
            Night.night
        )
        for n, t in Parallel(n_jobs=args.processes, return_as='generator')(delayed(add_observations)(tmin, tmax, exposure) for night, tmin, tmax, exposure in night_query):
            print(f"Number of missed observations: {n}/{t}")
            num_missed += n
            num_total += t
    print(f"Number of missed observations: {num_missed}/{num_total}")
            
if __name__ == "__main__":
    main()
