from sqlalchemy import Column, String, Integer, Float, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Observable(Base):
    """
    This is an observable of a moving object relative to a body in a provided frame
    It has a latitude and longitude, velocity, and uncertainty
    """
    __tablename__ = "observable"
    id = Column(Integer, primary_key=True, autoincrement=True)
    moving_object_id = Column(Integer, ForeignKey("moving_object.id"))
    # observable
    longitude = Column(Float)
    latitude = Column(Float)
    v_longitude = Column(Float)
    v_latitude = Column(Float)
    # uncertainty
    sigma_longitude = Column(Float)
    sigma_latitude = Column(Float)
    sigma_v_longitude = Column(Float)
    sigma_v_latitude = Column(Float)

    frame = Column(String)
    center = Column(String)

    moving_object = relationship("MovingObject", back_populates="observables")

class State(Base):
    """
    This is the state of a moving object relative to a body in a provided frame
    It has a position (x, y, z)
    It has a velocity (vx, vy, vz)
    It has an uncertainty (sigma_*)
    """
    __tablename__ = "state"
    id = Column(Integer, primary_key=True, autoincrement=True)
    moving_object_id = Column(Integer, ForeignKey("moving_object.id"), nullable=True)
    time = Column(Float)
    # cartesian position
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    vx = Column(Float)
    vy = Column(Float)
    vz = Column(Float)
    # uncertainties
    sigma_x = Column(Float)
    sigma_y = Column(Float)
    sigma_z = Column(Float)
    sigma_vx = Column(Float)
    sigma_vy = Column(Float)
    sigma_vz = Column(Float)

    frame = Column(String)
    center = Column(String)

    moving_object = relationship("MovingObject", back_populates="states")

class Element(Base):
    """
    This holds the parameters of a Keplerian orbit
    """
    __tablename__ = "element"
    id = Column(Integer, primary_key=True)
    moving_object_id = Column(Integer, ForeignKey("moving_object.id"), nullable=True)
    epoch_jd_tdb = Column(Float) # the epoch of the orbit in MJD format TDB scale
    a = Column(Float) # the semimajor axis
    q = Column(Float) # the perifocal distance
    e = Column(Float) # the eccentricity
    i = Column(Float) # the inclination
    node = Column(Float) # the longitude of ascending node
    argperi = Column(Float) # the argument of perihelion
    peri_time = Column(Float) # the time of periastron passage
    period = Column(Float) # the period of the orbit
    mean_anomaly = Column(Float) # the mean anomaly
    mean_motion = Column(Float) # the mean motion
    # uncertainties
    sigma_a = Column(Float)
    sigma_q = Column(Float)
    sigma_e = Column(Float)
    sigma_i = Column(Float)
    sigma_node = Column(Float)
    sigma_argperi = Column(Float)
    sigma_peri_time = Column(Float)
    sigma_period = Column(Float)
    sigma_mean_anomaly = Column(Float)
    sigma_mean_motion = Column(Float)

    moving_object = relationship("MovingObject", back_populates="elements")

class MovingObject(Base):
    """
    This is a moving object, which comes from an identity source
    It might be a previously known moving object (from the MPC)
    or a synthetic object.

    All moving objects have states, observables, and orbits
    """
    __tablename__ = "moving_object"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mpc_object_id = Column(Integer, ForeignKey("mpc_object.id"), nullable=True)
    synthetic_object_id = Column(Integer, ForeignKey("synthetic_object.id"), nullable=True)

    states = relationship("State")
    observables = relationship("Observable")
    elements = relationship("Element")
