import logging
logger = logging.getLogger(__name__)

from deep_db.visualize.common import _equalize_lims, _plot_detector_exposure_boxes, _filter_by_geometry


def _plot_observations(ax, fig, observations):
    import matplotlib.dates as mdates
    ra = [o.ra for o in observations]
    dec = [o.dec for o in observations]
    obstime = mdates.date2num([o.obstime for o in observations])
    points = ax.scatter(ra, dec, c=obstime)
    cbar = fig.colorbar(points, ax=ax, label="obstime")
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))


def _plot_observations_relative_hours(ax, fig, observations, label=None):
    ra = [o.ra for o in observations]
    dec = [o.dec for o in observations]
    obstime = [o.obstime for o in observations]
    t0 = min(obstime)
    hours = [(t - t0).total_seconds() / 3600.0 for t in obstime]
    points = ax.scatter(ra, dec, c=hours, label=label)
    fig.colorbar(points, ax=ax, label="obstime [hours since min(obstime)]")


def _plot_mpchecker_ephemeris(ax, ephemerides, label=None):
    ephemerides = sorted(ephemerides, key=lambda e: e.time)
    ax.plot([e.ra for e in ephemerides], [e.dec for e in ephemerides], color='k', lw=1, zorder=1, label=label)


def _plot_tracklet_arrow(ax, mpc_tracklet, color, alpha=1.0, linewidth=5, time_start=None, time_stop=None):
    from numpy import array
    obs = sorted(mpc_tracklet.observations, key=lambda x : x.obstime)
    t_start = obs[0].obstime
    t_stop = obs[-1].obstime
    # fit a line to the observations ra/dec
    ra_start, dec_start = line @ array([t_start, 1])
    ra_stop, dec_stop = line @ array([t_stop, 1])
    # annotate() arrows aren't included in axis autoscaling, so plot the
    # endpoints invisibly to make sure the arrow isn't clipped
    ax.plot([ra_start, ra_stop], [dec_start, dec_stop], alpha=0)
    ax.annotate(
        "",
        xy=(ra_stop, dec_stop),
        xytext=(ra_start, dec_start),
        arrowprops=dict(arrowstyle="-|>", color=color, alpha=alpha, linewidth=linewidth),
    )


def visualize_known_objects(
    session,
    provid=None, permid=None, name=None,
    night=None, field=None, expnum=None, detnum=None, detname=None,
    obs=True, eph=True, plot_ccds=False, annotate_ccds=False, output=None,
):
    """
    Query the known object(s) by provid/permid/name and/or by survey
    geometry (night/field/expnum/detnum/detname, joined in through
    Exposure/DetectorExposure -- see _filter_by_geometry). Both kinds of
    filter can be combined (all given conditions apply together, AND'ed):
    e.g. night=[20190402], field=["A0a"] plots the observations of every
    known object seen on that night/field regardless of designation, while
    provid=[...], night=[...] restricts one object's observations down to
    the given night(s).

    By default, plots both the ra/dec of matching mpc_observations
    (colored by obstime) and matching mpchecker_ephemeris (connected in
    time order), warning instead of silently producing an empty plot when
    either is unavailable. With plot_ccds=True, also outlines each
    distinct CCD pointing the observations landed on, colored/labeled by
    (night, field); annotate_ccds=True additionally labels each box with
    its detector.
    """
    import matplotlib.pyplot as plt
    from deep_db.database.mpc import MPCObject, MPCObservation
    from deep_db.database.mpchecker import MPCheckerEphemeris

    has_designation = bool(provid or permid or name)
    has_geometry = bool(night or field or expnum or detnum or detname)
    if not has_designation and not has_geometry:
        raise ValueError("Must specify at least one of provid/permid/name or night/field/expnum/detnum/detname")

    def _filter_by_designation(q, object_col):
        q = q.join(MPCObject, object_col == MPCObject.id)
        if provid:
            q = q.filter(MPCObject.provid.in_(provid))
        if permid:
            q = q.filter(MPCObject.permid.in_(permid))
        if name:
            q = q.filter(MPCObject.name.in_(name))
        return q

    obs_q = session.query(MPCObservation)
    if has_designation:
        obs_q = _filter_by_designation(obs_q, MPCObservation.object_id)
    if has_geometry:
        obs_q = _filter_by_geometry(
            obs_q, MPCObservation.exposure_id, MPCObservation.detector_exposure_id,
            night=night, field=field, expnum=expnum, detnum=detnum, detname=detname,
        )
    observations = obs_q.distinct().all()

    eph_q = session.query(MPCheckerEphemeris)
    if has_designation:
        eph_q = _filter_by_designation(eph_q, MPCheckerEphemeris.object_id)
    if has_geometry:
        eph_q = _filter_by_geometry(
            eph_q, MPCheckerEphemeris.exposure_id, MPCheckerEphemeris.detector_exposure_id,
            night=night, field=field, expnum=expnum, detnum=detnum, detname=detname,
        )
    ephemerides = eph_q.distinct().all()

    if has_designation and not observations and not ephemerides:
        # the object/night/field combination may just have no data (handled
        # below via the obs/eph warnings) -- only treat this as a hard
        # error if the designation itself doesn't resolve to any object
        exists_q = session.query(MPCObject)
        if provid:
            exists_q = exists_q.filter(MPCObject.provid.in_(provid))
        if permid:
            exists_q = exists_q.filter(MPCObject.permid.in_(permid))
        if name:
            exists_q = exists_q.filter(MPCObject.name.in_(name))
        if exists_q.first() is None:
            raise ValueError("No MPCObject found matching the given provid/permid/name")

    object_ids = {o.object_id for o in observations if o.object_id is not None}
    object_ids |= {e.object_id for e in ephemerides if e.object_id is not None}
    objects = session.query(MPCObject).filter(MPCObject.id.in_(object_ids)).all() if object_ids else []

    observations_by_object = {}
    for o in observations:
        observations_by_object.setdefault(o.object_id, []).append(o)
    ephemerides_by_object = {}
    for e in ephemerides:
        ephemerides_by_object.setdefault(e.object_id, []).append(e)

    fig, ax = plt.subplots(figsize=(4, 4))

    # all observations share one scatter call (and thus one colorbar/color
    # scale) regardless of how many objects matched -- plotting per-object
    # as before added a separate colorbar for every object, which was
    # unreadable once night/field selection started matching dozens of them
    if obs and observations:
        _plot_observations_relative_hours(ax, fig, observations, label="MPC Observations")
    if obs and not observations:
        logger.warning("No mpc_observation records found for the matching object(s)/geometry; skipping observations")

    # ephemeris is still plotted per-object (one connected line per object)
    # since combining different objects' points into a single line would
    # draw nonsensical connections between unrelated tracks
    if eph:
        for object in objects:
            object_ephemerides = ephemerides_by_object.get(object.id, [])
            if object_ephemerides:
                _plot_mpchecker_ephemeris(ax, object_ephemerides, label="Ephemeris")
    if eph and not ephemerides:
        logger.warning("No mpchecker_ephemeris records found for the matching object(s)/geometry; skipping ephemeris")

    if plot_ccds:
        detector_exposures = {
            o.detector_exposure_id: o.detector_exposure
            for o in [*observations, *ephemerides]
            if o.detector_exposure is not None
        }
        if detector_exposures:
            _plot_detector_exposure_boxes(ax, detector_exposures.values(), annotate=annotate_ccds)

    def _label(o):
        # some ingested rows store the literal string "None" instead of
        # a real NULL for missing name/permid, so filter those out too
        l = []
        if o.permid and o.permid != "None":
            l.append(o.permid)
        elif o.name and o.name != "None":
            l.append(o.name)
        if l:
            l.append("(" + o.provid + ")")
        else:
            l.append(o.provid)
        l = " ".join(l)
        return l

    ax.set_xlabel("ra [deg]")
    ax.set_ylabel("dec [deg]")
    title_parts = []
    if objects:
        title_parts.append(", ".join(_label(o) for o in objects))
    if night:
        title_parts.append(f"night {', '.join(map(str, night))}")
    if field:
        title_parts.append(f"field {', '.join(field)}")
    if expnum:
        title_parts.append(f"expnum {', '.join(map(str, expnum))}")
    if detnum:
        title_parts.append(f"detnum {', '.join(map(str, detnum))}")
    if detname:
        title_parts.append(f"detname {', '.join(detname)}")
    ax.set_title(" / ".join(title_parts))
    ax.set_aspect("equal")
    _equalize_lims(ax)
    ax.grid(True)
    ax.set_axisbelow(True)

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

def visualize_tracklets(session, trkid=None, trksub=None, obs=True, output=None):
    """
    Query the mpc_tracklet by trkid/trksub
    Plot the tracklet as an arrow from min(obs.obstime), max(obs.obstime)
    If obs, scatter the ra/dec of the associated mpc_observations
    """
    import matplotlib.pyplot as plt
    from deep_db.database.mpc import MPCTracklet
    q = session.query(MPCTracklet)
    if trkid:
        q = q.filter(MPCTracklet.trkid.in_(trkid))
    if trksub:
        q = q.filter(MPCTracklet.trksub.in_(trksub))

    fig = plt.figure()
    ax = fig.add_subplot()
    for tracklet in q.distinct():
        _plot_tracklet_arrow(ax, tracklet)
