"""
This is Pedro's fakes tables represented as database schema
"""
from sqlalchemy import Column, String, Integer, Float, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import declarative_base, relationship
from .models import Base

class SyntheticObjectDesignation(Base):
    """
    The identity of a synthetic (fake) object as a whole system, shared by
    every SyntheticObject component belonging to it (just the 'primary' for
    most fakes; 'primary' and 'companion' for binaries).

    `orbit_id` unifies ORBITID across the 'asteroid' and 'tno' fake-object
    populations into one space, following the DEEP fakes pipeline's own
    convention (seen in fakes_detections_joined.fits): the raw 'tno' ORBITID
    is kept as-is, the raw 'asteroid' ORBITID is offset by 10,000,000 --
    the two populations' raw ORBITID ranges otherwise overlap.
    """
    __tablename__ = 'synthetic_object_designation'
    __table_args__ = (
        UniqueConstraint("orbit_id", name="uq_synthetic_object_designation_orbit_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    orbit_id = Column(Integer, nullable=False)
    name = Column(String)

class SyntheticObject(Base):
    """
    A synthetic (fake) object injected into the survey by the DEEP fakes
    pipeline, one row per (source, orbit_id, component).

    ORBITID tracks a binary system's barycenter, not either individual
    body: a binary gets two rows sharing one ORBITID -- 'primary' and
    'companion' -- which also share one element_id/state_id, since there's
    only one integrated orbit/state (the barycenter's) to point at. Every
    non-binary ORBITID has only the 'primary' row.

    ORBITID is only unique within a `source` population (e.g. the 'asteroid'
    and 'tno' fake-object populations assign ORBITID from overlapping
    ranges), so the natural key is (source, orbit_id, component), not
    orbit_id alone.
    """
    __tablename__ = "synthetic_object"
    __table_args__ = (
        UniqueConstraint(
            "source", "orbit_id", "component",
            name="uq_synthetic_object_source_orbit_id_component",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    designation_id = Column(Integer, ForeignKey("synthetic_object_designation.id"))
    element_id = Column(Integer, ForeignKey("element.id"))
    state_id = Column(Integer, ForeignKey("state.id"))

    source = Column(String, nullable=False)
    orbit_id = Column(Integer, nullable=False) # this is the ORBITID column
    component = Column(String, nullable=False, default="primary")  # 'primary' or 'companion'

    n_images = Column(Integer)  # N_IMAGES: number of survey images the fake lands on
    m_vr = Column(Float)        # m_VR
    h_vr = Column(Float)        # H_VR: absolute magnitude (V-R band)
    d = Column(Float)
    r = Column(Float)
    amplitude = Column(Float)   # AMP: lightcurve amplitude (mag)
    phase = Column(Float)       # PHASE: rotational lightcurve phase offset
    period = Column(Float)      # PERIOD: rotation period; 0 where no lightcurve was simulated

    designation = relationship("SyntheticObjectDesignation")
    element = relationship("Element")
    state = relationship("State")

class SyntheticObjectBinary(Base):
    """
    Links a binary SyntheticObject pair -- primary and companion, same
    source/orbit_id -- and records the companion's properties relative to
    the primary (only a subset of synthetic objects have one of these).
    """
    __tablename__ = "synthetic_object_binary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    primary_synthetic_object_id = Column(Integer, ForeignKey("synthetic_object.id"), nullable=False, unique=True)
    companion_synthetic_object_id = Column(Integer, ForeignKey("synthetic_object.id"), nullable=False, unique=True)
    delta_h = Column(Float)    # DELTA_H: magnitude difference between primary and companion
    separation = Column(Float) # SEPARATION
    angle = Column(Float)      # ANGLE

    primary_synthetic_object = relationship("SyntheticObject", foreign_keys=[primary_synthetic_object_id])
    companion_synthetic_object = relationship("SyntheticObject", foreign_keys=[companion_synthetic_object_id])

class SyntheticObjectEphemeris(Base):
    """
    The predicted position of a SyntheticObject at a single Exposure's
    obstime, as produced by the DEEP fakes injection pipeline -- the
    synthetic-object analog of MPCheckerEphemeris. Defined by the exposure
    table of obstimes it's evaluated against -- exposure_id is a real,
    non-nullable FK, set regardless of where the position falls.
    detector_exposure_id is nullable: set when the position falls within
    one of that Exposure's DetectorExposures' bounds, left null when it
    misses all of them -- useful for knowing what missed the survey
    geometry. As with MPCheckerEphemeris, the evaluation time isn't stored
    directly; it's always `exposure`'s midpoint, read through the `time`
    property below.
    """
    __tablename__ = "synthetic_object_ephemeris"
    __table_args__ = (
        UniqueConstraint(
            "synthetic_object_id", "exposure_id", "detector_exposure_id",
            name="uq_synthetic_object_ephemeris_object_exposure_detector_exposure",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    synthetic_object_id = Column(Integer, ForeignKey("synthetic_object.id"), nullable=False)
    exposure_id = Column(Integer, ForeignKey("exposure.id"), nullable=False)
    detector_exposure_id = Column(Integer, ForeignKey("detector_exposure.id"), nullable=True)

    ra = Column(Float)
    dec = Column(Float)
    mag = Column(Float)  # MAG: apparent magnitude at this epoch
    d = Column(Float)
    r = Column(Float)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    vx = Column(Float)
    vy = Column(Float)
    vz = Column(Float)

    synthetic_object = relationship("SyntheticObject")
    exposure = relationship("Exposure")
    detector_exposure = relationship("DetectorExposure")

    @property
    def time(self):
        return self.exposure.midpoint
