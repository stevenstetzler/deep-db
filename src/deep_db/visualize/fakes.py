import logging
logger = logging.getLogger(__name__)

from deep_db.visualize.common import _equalize_lims, _plot_detector_exposure_boxes, _filter_by_geometry


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
