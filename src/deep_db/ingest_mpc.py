import logging
import sys
import argparse

from joblib import Parallel, delayed
from sqlalchemy import MetaData, Table, create_engine, func, insert, select, distinct
from sqlalchemy.orm import Session
import astropy.units as u
from .models import Base, Exposure, Night, MPCObservation, ra_dec_to_coordinate

logger = logging.getLogger(__name__)


def bulk_insert_observations(engine, observations, batch_size=1000):
    obsids = set()
    for row in observations:
        obsids.add(row['obsid'])

    with Session(engine) as deep_db:
        # find which ids already exist, insert only the rest
        existing = set()
        obsids_to_check = []
        for obsid in obsids:
            if len(obsids_to_check) == batch_size:
                existing.update(
                    set(
                        deep_db.scalars(
                            select(MPCObservation.id).where(MPCObservation.id.in_(obsids_to_check))
                        )
                    )
                )
                obsids_to_check = []
            obsids_to_check.append(obsid)
        if len(obsids_to_check) > 0:
            existing.update(
                set(
                    deep_db.scalars(
                        select(MPCObservation.id).where(MPCObservation.id.in_(obsids_to_check))
                    )
                )
            )
        new_ids = obsids - existing

        if not new_ids:
            logger.info("all %d MPC objects already present", len(obsids))
            return

        rows_to_insert = []
        for row in observations:
            if len(rows_to_insert) == batch_size:
                deep_db.execute(insert(MPCObservation), rows_to_insert)
                rows_to_insert = []
            if row['obsid'] in new_ids:
                row['id'] = row.pop("obsid")
                rows_to_insert.append(row)

        if len(rows_to_insert) > 0:
            deep_db.execute(insert(MPCObservation), rows_to_insert)
        deep_db.commit()

    logger.info("inserted %d new MPC observations (%d already present)",
                len(new_ids), len(existing))

def get_mpc_data(engine, tmin, tmax, stn="W84", metadata=None):
    """Fetch MPC-submitted data for a single night / block of observing.

    Parameters
    ----------
    engine : sqlalchemy.Engine
        Engine connected to the database containing ``obs_sbn``
        (e.g. ``mpc_engine``).
    tmin, tmax : datetime | float | str
        Bounds on ``obstime`` (exclusive on both ends, matching the
        original ``obstime > tmin AND obstime < tmax``).
    stn : str
        Observatory station code. Defaults to ``'W84'``.
    metadata : sqlalchemy.MetaData, optional
        Reuse an existing MetaData for table reflection. A new one is
        created if not supplied.

    Returns
    -------
    (objects, tracklets, observations) : tuple of dict

        observations : {obsid: {trkid, trksub, provid, permid,
                                obstime, ra, dec, stn, status}}
        tracklets    : {trkid: [obsid, ...]}
        objects      : {packed_primary_provisional_designation: [trkid, ...]}
    """
    if metadata is None:
        metadata = MetaData()

    def _reflect(name):
        # reuse an already-reflected table if present, else autoload it
        return metadata.tables.get(name) or Table(
            name, metadata, autoload_with=engine
        )

    obs_sbn = _reflect("obs_sbn")
    numbered_identifications = _reflect("numbered_identifications")
    current_identifications = _reflect("current_identifications")

    # shared time / station filter
    block_filter = (
        obs_sbn.c.stn == stn,
        obs_sbn.c.obstime > tmin,
        obs_sbn.c.obstime < tmax,
    )

    observations = []

    with Session(engine) as session:
        # --- observations -------------------------------------------------
        obs_stmt = select(
            obs_sbn.c.obsid,
            obs_sbn.c.trkid,
            obs_sbn.c.trksub,
            obs_sbn.c.obstime,
            obs_sbn.c.ra,
            obs_sbn.c.dec,
            obs_sbn.c.stn,
            obs_sbn.c.status,
            func.coalesce(
                numbered_identifications.c.packed_primary_provisional_designation,
                current_identifications.c.packed_primary_provisional_designation,
            ).label("packed_designation"),
            obs_sbn.c.provid,
            obs_sbn.c.permid,
        ).select_from(
            obs_sbn.outerjoin(
                numbered_identifications,
                obs_sbn.c.permid == numbered_identifications.c.permid,
            ).outerjoin(
                current_identifications,
                obs_sbn.c.provid
                == current_identifications.c.unpacked_secondary_provisional_designation,
            )
        ).where(*block_filter)

        for row in session.execute(obs_stmt).mappings():
            row = dict(row)
            ra, dec, hp_index = ra_dec_to_coordinate(row['ra'] * u.deg, row['dec'] * u.deg)
            row['ra'] = ra
            row['dec'] = dec
            row['hp_index'] = hp_index
            observations.append(row)

    return observations

def main():
    parser = argparse.ArgumentParser(description="Query MPC ephemeris data for Deep DB")
    parser.add_argument("--mpc", required=True, help="MPC ephemeris data URL (PostgreSQL)")
    parser.add_argument("--db", required=True, help="Database URL")
    parser.add_argument("--processes", "-J", type=int, default=1, help="Number of processes to use")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements")
    parser.add_argument("--no-hp-prefilter", action="store_true",
                        help="Disable HEALPix candidate prefilter (full detector-exposure scan per row)")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    use_hp_prefilter = not args.no_hp_prefilter

    # mpc_engine = create_engine(args.mpc, echo=args.echo)
    deep_engine = create_engine(args.db, echo=args.echo)
    Base.metadata.create_all(deep_engine)

    night_data = []
    with Session(deep_engine) as deep_db:
        q = deep_db.query(
            Night.night,
            func.min(Exposure.obstime),
            func.max(Exposure.obstime),
            func.max(Exposure.exposure),
        ).join(Exposure).group_by(Night.night).order_by(Night.night)
        for night, tmin, tmax, _ in q:
            night_data.append((night, tmin, tmax))

    def _get_mpc_data(tmin, tmax):
        engine = create_engine(args.mpc, echo=args.echo)
        return get_mpc_data(engine, tmin, tmax)

    mpc_data = sum(Parallel(n_jobs=args.processes)(delayed(_get_mpc_data)(tmin, tmax) for _, tmin, tmax in night_data), [])

    objects = set(list(map(lambda x : x['packed_designation'], mpc_data)))
    print("inserting observations for", len(objects), "objects")

    bulk_insert_observations(deep_engine, mpc_data)

    # mpc_objects = list(map(lambda x : x[0], mpc_data))
    # mpc_tracklets = list(map(lambda x : x[1], mpc_data))
    # mpc_observations = list(map(lambda x : x[2], mpc_data))

    # bulk_insert_objects(deep_engine, mpc_objects)


if __name__ == "__main__":
    main()