"""
This defines the survey as a database schema
"""

from sqlalchemy import Integer, Column, ForeignKey, String, Float, UniqueConstraint, JSON, DateTime, Index
from sqlalchemy.orm import relationship, declarative_base
from astropy_healpix import HEALPix
import astropy.units as u
from astropy.coordinates import SkyCoord
from deep_db.database.models import Base
from deep_db.util.decam import ccdBounds, ccd_name_to_num, ccd_num_to_name
from deep_db.util.coord import sky_to_gnomonic

hp = HEALPix(nside=256, order='nested', frame='icrs')

def ra_dec_to_coordinate(ra, dec):
    hp_index = hp.lonlat_to_healpix(ra, dec)
    coord = SkyCoord(ra=ra, dec=dec, frame='icrs')
    return float(coord.ra.to(u.deg).value), float(coord.dec.to(u.deg).value), int(hp_index)

# Observation model
class Detector(Base):
    __tablename__ = 'detector'
    id = Column(Integer, primary_key=True)
    number = Column(Integer)
    name = Column(String)
    detector_exposures = relationship("DetectorExposure")

class Exposure(Base):
    __tablename__ = 'exposure'
    id = Column(Integer, primary_key=True)
    night_id = Column(Integer, ForeignKey('night.id'), nullable=False, index=True)
    field_id = Column(Integer, ForeignKey('field.id'), nullable=True, index=True)

    expnum = Column(Integer, unique=True)
    caldat = Column(String)
    exposure_time = Column(Float)
    obstime = Column(DateTime)
    mjd = Column(Float)
    midpoint = Column(DateTime)
    midpoint_mjd = Column(Float)
    band = Column(String, index=True)
    target = Column(String, index=True)
    ra = Column(Float)
    dec = Column(Float)
    hp_index = Column(Integer, index=True)

    night = relationship("Night", back_populates="exposures")
    field = relationship("Field", back_populates="exposures")
    detector_exposures = relationship("DetectorExposure")

    def contains(self, ra, dec):
        from astropy.units import deg
        return SkyCoord(ra, dec).separation(SkyCoord(self.ra * deg, self.dec * deg)) < 1.2 * u.deg

    @property
    def endtime(self):
        from astropy.time import Time, TimeDelta
        return (Time(self.obstime) + TimeDelta(self.exposure_time, format='sec')).utc.datetime

class DetectorExposure(Base):
    __tablename__ = 'detector_exposure'
    id = Column(Integer, primary_key=True)
    detector_id = Column(Integer, ForeignKey('detector.id'), nullable=False, index=True)
    exposure_id = Column(Integer, ForeignKey('exposure.id'), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint('detector_id', 'exposure_id', name='detector_exposure_uc'),
    )

    # center
    ra = Column(Float)
    dec = Column(Float)
    hp_index = Column(Integer, index=True)

    # corners
    ra_00 = Column(Float)
    dec_00 = Column(Float)
    hp_index_00 = Column(Integer, index=True)

    ra_01 = Column(Float)
    dec_01 = Column(Float)
    hp_index_01 = Column(Integer, index=True)

    ra_10 = Column(Float)
    dec_10 = Column(Float)
    hp_index_10 = Column(Integer, index=True)

    ra_11 = Column(Float)
    dec_11 = Column(Float)
    hp_index_11 = Column(Integer, index=True)

    detector = relationship("Detector", back_populates="detector_exposures")
    exposure = relationship("Exposure", back_populates="detector_exposures")

    def contains(self, ra, dec):
        ra, dec, _ = ra_dec_to_coordinate(ra, dec)
        xi, eta = sky_to_gnomonic(ra, dec, self.exposure.ra, self.exposure.dec)
        xmin, xmax, ymin, ymax = ccdBounds[ccd_num_to_name[self.detector.number]]
        return xmin <= xi <= xmax and ymin <= eta <= ymax

class Night(Base):
    __tablename__ = 'night'
    id = Column(Integer, primary_key=True)
    night = Column(Integer, unique=True, index=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    exposures = relationship("Exposure", back_populates="night")

class Field(Base):
    __tablename__ = 'field'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    exposures = relationship("Exposure", back_populates="field")
