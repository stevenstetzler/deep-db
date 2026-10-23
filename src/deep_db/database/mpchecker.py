from sqlalchemy import Column, String, Integer, Float, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import declarative_base, relationship
from deep_db.database.models import Base

class MPCheckerEphemeris(Base):
    """
    The ephemeris predicted by jOrbit.mpchecker for an mpc_designation.

    Defined by an Exposure's field of view and observation time --
    exposure_id is a real, non-nullable FK, set for every prediction in
    that field of view regardless of where it falls. detector_exposure_id
    is nullable: set when the predicted position falls within one of that
    Exposure's DetectorExposures' bounds, left null when it misses all of
    them (still in the wide field of view mpchecker was queried over, just
    not on any detector) -- useful for knowing what missed the survey
    geometry.
    """
    __tablename__ = "mpchecker_ephemeris"
    __table_args__ = (
        UniqueConstraint(
            "object_id", "exposure_id", "detector_exposure_id",
            name="uq_mpchecker_ephemeris_designation_exposure_detector_exposure",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    object_id = Column(Integer, ForeignKey("mpc_object.id"), nullable=True)
    exposure_id = Column(Integer, ForeignKey("exposure.id"), nullable=False)
    detector_exposure_id = Column(Integer, ForeignKey("detector_exposure.id"), nullable=True)
    
    ra = Column(Float)
    dec = Column(Float)
    v_mag = Column(Float)

    exposure = relationship("Exposure")
    detector_exposure = relationship("DetectorExposure")

    @property
    def time(self):
        """
        The ephemeris is always evaluated at its Exposure's midpoint, so
        that time is read through `exposure` instead of being duplicated
        in this table -- exposure_id (unlike detector_exposure_id) is
        always set, so this works for a row that missed the survey
        geometry too.
        """
        return self.exposure.midpoint
