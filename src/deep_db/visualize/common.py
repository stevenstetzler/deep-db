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
