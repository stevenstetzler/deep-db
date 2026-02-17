import logging
import os
import sys
import argparse
from datetime import timedelta
from random import random
from time import sleep

import astropy.units as u
from joblib import Parallel, delayed
from sqlalchemy import MetaData, Table, create_engine, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import (Base, DetectorExposure, Exposure, MPCObservation,
                     MPCTracklet, Night, SolarSystemObject,
                     ra_dec_to_coordinate)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Query MPC ephemeris data for Deep DB")
    parser.add_argument("--mpc", required=True, help="MPC ephemeris data URL")
    parser.add_argument("--db", required=True, help="Database URL")
    parser.add_argument("--processes", "-J", type=int, default=1, help="Number of processes to use")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements")
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

    deep_engine = create_engine(args.db, echo=args.echo)
    Base.metadata.create_all(deep_engine)

    mpc_engine = create_engine(args.mpc, echo=args.echo)
    metadata = MetaData()
    metadata.reflect(bind=mpc_engine)
    obs_sbn = Table("obs_sbn", metadata, autoload_with=mpc_engine)

    mpc_columns = ["trkid", "trksub", "permid", "provid", "ra", "dec", "mag", "band", "obstime", "status"]
    select = list(map(lambda x : getattr(obs_sbn.columns, x), mpc_columns))

    def query_objects(tmin, tmax, exposure):
        provids = set()
        
        mpc_engine = create_engine(args.mpc)
        with Session(mpc_engine) as mpc:
            end_time = tmax + timedelta(seconds=exposure)
            logger.debug(f"Querying MPC for provids between {tmin} and {end_time}")
            mpc_query = mpc.query(
                obs_sbn.columns.provid
            ).filter(
                obs_sbn.columns.stn == "W84",
                obs_sbn.columns.obstime >= tmin,
                obs_sbn.columns.obstime <= end_time
            ).distinct()

            for (provid,) in mpc_query:
                if provid is not None:
                    logger.debug(f"Found provid: {provid}")
                    provids.add(provid)

        logger.info(f"Retrieved {len(provids)} unique provids for time window.")
        return provids

    def add_tracklets(tmin, tmax, exposure):
        deep_engine = create_engine(args.db)
        mpc_engine = create_engine(args.mpc)
        with Session(deep_engine) as deep_db, Session(mpc_engine) as mpc:
            endtime = tmax + timedelta(seconds=exposure)
            logger.info(f"querying for tracklets between {tmin} and {endtime}")
            mpc_rows = mpc.query(
                obs_sbn.columns.provid,
                obs_sbn.columns.trksub, 
                obs_sbn.columns.trkid
            ).filter(
                obs_sbn.columns.stn == "W84"
            ).filter(
                obs_sbn.columns.obstime > tmin
            ).filter(
                obs_sbn.columns.obstime < endtime
            ).group_by(
                obs_sbn.columns.provid,
                obs_sbn.columns.trksub, 
                obs_sbn.columns.trkid
            ).all()
            
            unique_provids = {r.provid for r in mpc_rows if r.provid}
            
            provid_to_id = {}
            if unique_provids:
                logger.debug(f"querying for objects associated with provids of tracklets")
                obj_q = deep_db.query(SolarSystemObject.name, SolarSystemObject.id).filter(
                    SolarSystemObject.name.in_(unique_provids),
                    SolarSystemObject.type == "mpc"
                )
                provid_to_id = {name: oid for name, oid in obj_q}
            
            logger.debug(f"querying for existing tracklets")
            existing_tracklets = set()
            existing_q = deep_db.query(
                MPCTracklet.object_id, MPCTracklet.trksub, MPCTracklet.trkid
            ).filter(
                MPCTracklet.object_id.in_(provid_to_id.values())
            )
            existing_tracklets = {(t.object_id, t.trksub, t.trkid) for t in existing_q}
            
            for (provid, trksub, trkid) in mpc_rows:
                obj_id = provid_to_id.get(provid)
                if (obj_id, trksub, trkid) not in existing_tracklets:
                    logger.debug(f"adding tracklet {provid}, {trksub}, {trkid}")
                    new_tracklet = MPCTracklet(
                        object_id=obj_id,
                        trksub=trksub,
                        trkid=trkid,
                    )
                    deep_db.add(new_tracklet)
                    existing_tracklets.add((obj_id, trksub, trkid))
            deep_db.commit()

    def add_observations(tmin, tmax, exposure):
        current_pid = os.getpid()
        num_missed = 0
        num_total = 0

        deep_engine = create_engine(args.db)
        mpc_engine = create_engine(args.mpc)
        with Session(deep_engine) as deep_db, Session(mpc_engine) as mpc:
            mpc_rows = mpc.query(*select).filter(
                obs_sbn.columns.stn == "W84",
                obs_sbn.columns.obstime > tmin,
                obs_sbn.columns.obstime < tmax + timedelta(seconds=exposure)
            ).all()
            if not mpc_rows:
                return 0, 0

            provids = {r.provid for r in mpc_rows if r.provid}
            obj_map = {
                name: oid 
                for name, oid in deep_db.query(
                    SolarSystemObject.name, SolarSystemObject.id
                ).filter(
                    SolarSystemObject.name.in_(provids)
                ).all()
            }

            tracklet_q = deep_db.query(
                MPCTracklet.id, 
                MPCTracklet.object_id, 
                MPCTracklet.trksub, 
                MPCTracklet.trkid
            ).filter(
                MPCTracklet.object_id.in_(obj_map.values())
            ).all()
            tracklet_map = {(t.object_id, t.trksub, t.trkid): t.id for t in tracklet_q}

            de_list = deep_db.query(DetectorExposure).join(Exposure).filter(
                Exposure.obstime >= tmin - timedelta(minutes=10),
                Exposure.obstime <= tmax + timedelta(seconds=exposure) + timedelta(minutes=10)
            ).all()

            existing_obs = set(
                deep_db.query(
                    MPCObservation.tracklet_id, 
                    MPCObservation.detector_exposure_id
                ).filter(
                    MPCObservation.tracklet_id.in_([t.id for t in tracklet_q])
                ).all()
            )

            for row in mpc_rows:
                num_total += 1
                row_dict = {c: v for c, v in zip(mpc_columns, row)}
                
                oid = obj_map.get(row_dict['provid'])
                if row_dict['provid'] and not oid:
                    num_missed += 1
                    continue

                tracklet_id = tracklet_map.get((oid, row_dict['trksub'], row_dict['trkid']))
                if not tracklet_id:
                    num_missed += 1
                    continue

                ra, dec, hp_index = ra_dec_to_coordinate(row_dict['ra'] * u.deg, row_dict['dec'] * u.deg)
                match_de = None
                for de in de_list:
                    # Basic spatial + time check
                    if (de.ra_00 <= ra <= de.ra_11 and 
                        de.dec_00 <= dec <= de.dec_11 and
                        de.exposure.obstime <= row_dict['obstime'] <= de.exposure.obstime + timedelta(seconds=de.exposure.exposure)):
                        match_de = de
                        break
                
                if not match_de:
                    num_missed += 1
                    continue

                if (tracklet_id, match_de.id) in existing_obs:
                    continue

                logger.debug(f"Adding obs for tracklet {tracklet_id}")
                new_obs = MPCObservation(
                    tracklet_id=tracklet_id,
                    detector_exposure_id=match_de.id,
                    ra=ra,
                    dec=dec,
                    obstime=row_dict['obstime'],
                    hp_index=hp_index,
                    band=row_dict['band'],
                    mag=row_dict['mag'],
                    status=row_dict['status'],
                )
                deep_db.add(new_obs)
                existing_obs.add((tracklet_id, match_de.id))

            deep_db.commit()
        return num_missed, num_total

    num_missed = 0
    num_total = 0
    with Session(deep_engine) as deep_db:
        night_data = deep_db.query(
            Night.night,
            func.min(Exposure.obstime),
            func.max(Exposure.obstime),
            func.max(Exposure.exposure)
        ).join(Exposure).group_by(Night.night).order_by(Night.night).all()

        # logger.info(f"Starting parallel object query with {args.processes} processes")
        # provids = Parallel(n_jobs=args.processes)(
        #     delayed(query_objects)(tmin, tmax, exp) for _, tmin, tmax, exp in night_data
        # )
        # provids = set().union(*provids)
        # for provid in provids:
        #     logger.debug(f"querying object {provid}")
        #     obj = deep_db.query(SolarSystemObject).filter_by(name=provid, type='mpc').first()
        #     if obj is None:
        #         logger.debug(f"adding object {provid}")
        #         obj = SolarSystemObject(name=provid, type='mpc')
        #         deep_db.add(obj)
        #     deep_db.commit()
        
        # logger.info(f"Starting parallel tracklet query with {args.processes} processes")
        # Parallel(n_jobs=args.processes)(
        #     delayed(add_tracklets)(tmin, tmax, exp) for _, tmin, tmax, exp in night_data
        # )
        
        logger.info(f"Starting parallel observation query {args.processes} processes")
        results = Parallel(n_jobs=args.processes, return_as='generator')(
            delayed(add_observations)(tmin, tmax, exp) for _, tmin, tmax, exp in night_data
        )
        
        for n, t in results:
            logger.info(f"Night Batch Complete - Missed: {n}/{t}")
            num_missed += n
            num_total += t

    logger.info(f"FINAL STATS: {num_missed}/{num_total} observations missed.")
            
if __name__ == "__main__":
    main()