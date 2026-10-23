import logging
logger = logging.getLogger(__name__)

def _equalize_lims(ax):
    """
    Expand the shorter of xlim/ylim around its center so both axes span the
    same range -- set_aspect("equal") only matches the on-screen scale (1
    unit in x drawn the same as 1 unit in y), it doesn't touch the data
    limits themselves, so a plot with a wide ra range and narrow dec range
    still ends up a tall, mostly-empty box rather than a filled square.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = max(x1 - x0, y1 - y0)
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    ax.set_xlim(xc - span / 2, xc + span / 2)
    ax.set_ylim(yc - span / 2, yc + span / 2)


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


def _plot_detector_exposure_box(ax, detector_exposure, color="0.5", alpha=0.5, linewidth=1, label=None):
    # walk the corners around the perimeter (00 -> 01 -> 11 -> 10 -> 00) so
    # the box doesn't self-intersect, then close the loop
    ra = [detector_exposure.ra_00, detector_exposure.ra_01, detector_exposure.ra_11, detector_exposure.ra_10, detector_exposure.ra_00]
    dec = [detector_exposure.dec_00, detector_exposure.dec_01, detector_exposure.dec_11, detector_exposure.dec_10, detector_exposure.dec_00]
    ax.plot(ra, dec, color=color, alpha=alpha, linewidth=linewidth, label=label)


def _annotate_detector_exposure_box(ax, detector_exposure, color="0.5"):
    ras = [detector_exposure.ra_00, detector_exposure.ra_01, detector_exposure.ra_10, detector_exposure.ra_11]
    decs = [detector_exposure.dec_00, detector_exposure.dec_01, detector_exposure.dec_10, detector_exposure.dec_11]
    ra_min, ra_max = min(ras), max(ras)
    dec_min, dec_max = min(decs), max(decs)
    # inset from the bottom-left corner so the label sits inside the box
    ra_pad = 0.05 * (ra_max - ra_min)
    dec_pad = 0.05 * (dec_max - dec_min)
    detector = detector_exposure.detector
    ax.text(
        ra_min + ra_pad, dec_min + dec_pad,
        f"({detector.number} / {detector.name})",
        ha="left", va="bottom", fontsize=6, color=color,
    )

def _plot_detector_exposure_boxes(ax, detector_exposures, annotate=False):
    """
    Plot a box for each distinct CCD pointing in detector_exposures (an
    iterable of DetectorExposure), de-duplicated by central (ra, dec) --
    e.g. repeat exposures of the same field land on the exact same
    pointing and would otherwise draw/annotate the same box multiple
    times. Boxes are colored by (Night.night, Field.name), with one
    legend entry per distinct (night, field) pair; annotate=True also
    labels each box with its (detector_number / detector_name).
    """
    import matplotlib.pyplot as plt

    unique_des = {}
    for de in detector_exposures:
        unique_des.setdefault((de.ra, de.dec), de)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors_by_night_field = {}
    for de in unique_des.values():
        exposure = de.exposure
        key = (exposure.night.night if exposure.night else None, exposure.field.name if exposure.field else None)
        if key not in colors_by_night_field:
            colors_by_night_field[key] = color_cycle[len(colors_by_night_field) % len(color_cycle)]

    labeled = set()
    for de in unique_des.values():
        exposure = de.exposure
        key = (exposure.night.night if exposure.night else None, exposure.field.name if exposure.field else None)
        color = colors_by_night_field[key]
        label = f"({key[0]} / {key[1]})"
        _plot_detector_exposure_box(ax, de, color=color, label=None if label in labeled else label)
        labeled.add(label)
        if annotate:
            _annotate_detector_exposure_box(ax, de, color=color)

    if colors_by_night_field:
        ax.legend(fontsize=6, loc="best")


def _unwrap_ra_func(ref_ra):
    """
    SyntheticObjectEphemeris.ra comes straight from the DEEP fakes FITS
    tables with no wrap-to-[0,360) normalization (unlike every other
    ra/dec in this schema) -- for a field near RA=0 that means detections
    at e.g. ra=-5.7 right next to a DetectorExposure box at ra=354.3 for
    the *same* patch of sky. Build a function that re-wraps any ra into
    degrees-from-ref (ref_ra = the first ephemeris point plotted), so
    everything plotted through it lands on the same number line.
    """
    def unwrap_ra(ra):
        return ra - 360.0 if ra - ref_ra > 180.0 else (ra + 360.0 if ra - ref_ra < -180.0 else ra)
    return unwrap_ra


class _UnwrappedRaDetectorExposure:
    """
    Read-only view of a DetectorExposure with every ra/ra_* attribute
    passed through unwrap_ra, so a box can be plotted on the same
    unwrapped number line as SyntheticObjectEphemeris points (see
    _unwrap_ra_func).
    """
    def __init__(self, detector_exposure, unwrap_ra):
        self._de = detector_exposure
        self.ra = unwrap_ra(detector_exposure.ra)
        self.dec = detector_exposure.dec
        for attr in ("ra_00", "ra_01", "ra_10", "ra_11"):
            setattr(self, attr, unwrap_ra(getattr(detector_exposure, attr)))
        for attr in ("dec_00", "dec_01", "dec_10", "dec_11"):
            setattr(self, attr, getattr(detector_exposure, attr))

    @property
    def detector(self):
        return self._de.detector

    @property
    def exposure(self):
        return self._de.exposure


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


def _filter_by_geometry(q, exposure_col, detector_exposure_col, night=None, field=None, expnum=None, detnum=None, detname=None):
    """
    Restrict a query of a row that carries its own exposure_id/
    detector_exposure_id (MPCObservation, MPCheckerEphemeris,
    SyntheticObjectEphemeris) down to the given survey geometry, joined in
    through Exposure/DetectorExposure -- see query.query_survey for the
    same join shape.
    """
    from deep_db.database.survey import Exposure, Night, Field, DetectorExposure, Detector

    q = q.join(Exposure, exposure_col == Exposure.id)
    if night:
        q = q.join(Night, Exposure.night_id == Night.id).filter(Night.night.in_(night))
    if field:
        q = q.join(Field, Exposure.field_id == Field.id).filter(Field.name.in_(field))
    if expnum:
        q = q.filter(Exposure.expnum.in_(expnum))
    if detnum or detname:
        # join on the row's own detector_exposure_id -- the specific CCD
        # it landed on -- not exposure_col == DetectorExposure.exposure_id,
        # which would match any CCD imaged in the same Exposure (i.e.
        # nearly all of them, since one Exposure covers ~60 CCDs at once)
        # regardless of where the row itself actually fell
        q = q.join(DetectorExposure, detector_exposure_col == DetectorExposure.id)
        q = q.join(Detector, Detector.id == DetectorExposure.detector_id)
        if detnum:
            q = q.filter(Detector.number.in_(detnum))
        if detname:
            q = q.filter(Detector.name.in_(detname))
    return q


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
        

def visualize_synthetic_objects(
    session, orbit_id=None, name=None,
    night=None, field=None, expnum=None, detnum=None, detname=None,
    plot_ccds=False, annotate_ccds=False, output=None,
):
    """
    Query synthetic (fake) object(s) by orbit_id/name
    (SyntheticObjectDesignation) and/or by survey geometry
    (night/field/expnum/detnum/detname, joined in through
    Exposure/DetectorExposure -- see _filter_by_geometry). Both kinds of
    filter combine (all given conditions apply together, AND'ed): e.g.
    night=[20190402], field=["A0a"] plots every synthetic object's
    ephemeris on that night/field regardless of designation, while
    orbit_id=[...], night=[...] restricts one object's ephemeris down to
    the given night(s).

    Scatters the ra/dec of each matching SyntheticObjectEphemeris
    (predicted per-exposure position), colored by time -- every 'primary'
    component shares one scatter call/colorbar regardless of how many
    objects matched (same reasoning as visualize_known_objects: separate
    per-object scatter calls each get their own independently-normalized
    color scale, which stops being readable once a geometry query starts
    matching many objects). A binary's 'companion' component is overlaid as
    black x markers, one legend entry each. With plot_ccds=True, also
    outlines each distinct CCD pointing the ephemerides landed on,
    colored/labeled by (night, field); annotate_ccds=True additionally
    labels each box with its detector.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from deep_db.database.synthetic_objects import SyntheticObjectDesignation, SyntheticObject, SyntheticObjectEphemeris

    has_designation = bool(orbit_id or name)
    has_geometry = bool(night or field or expnum or detnum or detname)
    if not has_designation and not has_geometry:
        raise ValueError("Must specify at least one of orbit_id/name or night/field/expnum/detnum/detname")

    eph_q = session.query(SyntheticObjectEphemeris)
    if has_designation:
        eph_q = eph_q.join(SyntheticObject, SyntheticObjectEphemeris.synthetic_object_id == SyntheticObject.id)
        eph_q = eph_q.join(SyntheticObjectDesignation, SyntheticObject.designation_id == SyntheticObjectDesignation.id)
        if orbit_id:
            eph_q = eph_q.filter(SyntheticObjectDesignation.orbit_id.in_(orbit_id))
        if name:
            eph_q = eph_q.filter(SyntheticObjectDesignation.name.in_(name))
    if has_geometry:
        eph_q = _filter_by_geometry(
            eph_q, SyntheticObjectEphemeris.exposure_id, SyntheticObjectEphemeris.detector_exposure_id,
            night=night, field=field, expnum=expnum, detnum=detnum, detname=detname,
        )
    ephemerides = eph_q.distinct().all()

    if has_designation and not ephemerides:
        # the designation/geometry combination may just have no data in that
        # window -- only a hard error if the designation itself doesn't
        # resolve to any SyntheticObjectDesignation at all
        exists_q = session.query(SyntheticObjectDesignation)
        if orbit_id:
            exists_q = exists_q.filter(SyntheticObjectDesignation.orbit_id.in_(orbit_id))
        if name:
            exists_q = exists_q.filter(SyntheticObjectDesignation.name.in_(name))
        if exists_q.first() is None:
            raise ValueError("No SyntheticObjectDesignation found matching the given orbit_id/name")

    object_ids = {e.synthetic_object_id for e in ephemerides}
    objects = session.query(SyntheticObject).filter(SyntheticObject.id.in_(object_ids)).all() if object_ids else []

    designation_ids = {o.designation_id for o in objects if o.designation_id is not None}
    designations = session.query(SyntheticObjectDesignation).filter(
        SyntheticObjectDesignation.id.in_(designation_ids)
    ).all() if designation_ids else []

    ephemerides_by_object = {}
    for e in ephemerides:
        ephemerides_by_object.setdefault(e.synthetic_object_id, []).append(e)
    for rows in ephemerides_by_object.values():
        rows.sort(key=lambda e: e.time)

    if not ephemerides:
        logger.warning("No synthetic_object_ephemeris records found for the matching object(s)/geometry")

    ref_ra = ephemerides[0].ra if ephemerides else 0.0
    unwrap_ra = _unwrap_ra_func(ref_ra)

    fig, ax = plt.subplots(figsize=(4, 4))

    if plot_ccds:
        detector_exposures = {
            e.detector_exposure_id: e.detector_exposure
            for e in ephemerides
            if e.detector_exposure is not None
        }
        if detector_exposures:
            unwrapped_des = [_UnwrappedRaDetectorExposure(de, unwrap_ra) for de in detector_exposures.values()]
            _plot_detector_exposure_boxes(ax, unwrapped_des, annotate=annotate_ccds)

    # all 'primary' components share one scatter call (and thus one
    # colorbar/color scale) -- see the docstring note on why
    primary_ra, primary_dec, primary_times = [], [], []
    for obj in objects:
        if obj.component != "primary":
            continue
        for e in ephemerides_by_object.get(obj.id, []):
            primary_ra.append(unwrap_ra(e.ra))
            primary_dec.append(e.dec)
            primary_times.append(e.time)

    points = None
    if primary_ra:
        times = mdates.date2num(primary_times)
        points = ax.scatter(primary_ra, primary_dec, c=times)

    for obj in objects:
        if obj.component == "primary":
            continue
        obj_ephemerides = ephemerides_by_object.get(obj.id, [])
        if not obj_ephemerides:
            continue
        ra = [unwrap_ra(e.ra) for e in obj_ephemerides]
        dec = [e.dec for e in obj_ephemerides]
        ax.scatter(ra, dec, color="k", marker="x", s=15, label=f"{obj.component} (SyntheticObject {obj.id})")

    if points is not None:
        cbar = fig.colorbar(points, ax=ax, label="time")
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    if any(obj.component != "primary" for obj in objects):
        ax.legend(fontsize=6, loc="best")

    ax.set_xlabel("ra [deg]")
    ax.set_ylabel("dec [deg]")
    title_parts = []
    if designations:
        title_parts.append(", ".join(d.name or str(d.orbit_id) for d in designations))
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
    ax.set_title("SyntheticObjectEphemeris for " + " / ".join(title_parts) if title_parts else "SyntheticObjectEphemeris")
    ax.set_aspect("equal")
    _equalize_lims(ax)
    ax.grid(True)
    ax.set_axisbelow(True)

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

