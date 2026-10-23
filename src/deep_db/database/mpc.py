"""
This is a mirror of the MPC SBN tables
"""
from sqlalchemy import Column, String, Integer, Float, ForeignKey, UniqueConstraint, CheckConstraint, DateTime
from sqlalchemy.orm import declarative_base, relationship
from .models import Base

class MPCObject(Base):
    """
    These are the known objects in the solar system
    """
    __tablename__ = "mpc_object"

    __table_args__ = (
        # one row per real object: provid/permid/name each identify it on
        # their own (see _resolve_designations), so a joint
        # (provid, permid, name) constraint doesn't stop two rows sharing
        # e.g. the same permid with a different provid/name from being
        # treated as different objects. NULL still isn't unique against
        # NULL under standard SQL semantics, so permid/name (nullable) can
        # each repeat as NULL across many rows -- only a non-NULL value is
        # required to be one-per-row.
        UniqueConstraint("provid", name="uq_mpc_object_provid"),
        UniqueConstraint("permid", name="uq_mpc_object_permid"),
        UniqueConstraint("name", name="uq_mpc_object_name"),
        CheckConstraint(
            "name IS NULL OR permid IS NOT NULL",
            name="ck_mpc_designation_name_requires_permanent",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    provid = Column(String, nullable=False) # unpacked provisional designation
    permid = Column(String, nullable=True) # permanent designation
    name = Column(String, nullable=True) # name
    
    observations = relationship("MPCObservation")
    tracklets = relationship("MPCTracklet", back_populates="object")

class MPCTracklet(Base):
    """
    An observation tracklet in the MPC obs_sbn database; keyed by trkid
    """
    __tablename__ = "mpc_tracklet"
    __table_args__ = (
        UniqueConstraint("trkid", "trksub", name="uq_mpc_tracklet_trkid_trksub"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    object_id = Column(Integer, ForeignKey("mpc_object.id"), nullable=True)
    
    trkid = Column(String)
    trksub = Column(String)

    object = relationship("MPCObject", back_populates="tracklets")
    observations = relationship("MPCObservation")

class MPCObservation(Base):
    __tablename__ = "mpc_observation"

    obsid = Column(String, primary_key=True)
    object_id = Column(Integer, ForeignKey("mpc_object.id"), nullable=True)
    exposure_id = Column(Integer, ForeignKey("exposure.id"), nullable=False)
    detector_exposure_id = Column(Integer, ForeignKey("detector_exposure.id"), nullable=True)
    mpc_tracklet_id = Column(Integer, ForeignKey("mpc_tracklet.id"), nullable=False)
    
    ra = Column(Float)
    dec = Column(Float)
    mag = Column(Float)
    obstime = Column(DateTime)
    status = Column(String, nullable=True)

    exposure = relationship("Exposure")
    detector_exposure = relationship("DetectorExposure")
    mpc_tracklet = relationship("MPCTracklet", back_populates="observations")
