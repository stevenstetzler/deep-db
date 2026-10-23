"""
This defines the survey as a database schema
"""

from sqlalchemy import Integer, Column, ForeignKey, String, Float, UniqueConstraint, JSON, DateTime, Index
from sqlalchemy.orm import relationship, declarative_base
from astropy_healpix import HEALPix
import astropy.units as u
from astropy.coordinates import SkyCoord
from deep_db.database.models import Base

"""
Taken from Bernardinelli DESTNOSIM https://github.com/bernardinelli/DESTNOSIM/blob/master/destnosim/des/ccd.py
"""

ccdBounds = {'N1': (-1.0811, -0.782681, -0.157306, -0.00750506),
             'N2': (-0.771362, -0.472493, -0.157385, -0.00749848), 
             'N3': (-0.461205, -0.161464, -0.157448, -0.00749265), 
             'N4': (-0.150127, 0.149894, -0.15747, -0.00749085), 
             'N5': (0.161033, 0.460796, -0.157638, -0.0074294), 
             'N6': (0.472171, 0.771045, -0.157286, -0.00740563), 
             'N7': (0.782398, 1.08083, -0.157141, -0.0074798), 
             'N8': (-0.92615, -0.627492, -0.321782, -0.172004), 
             'N9': (-0.616455, -0.317043, -0.322077, -0.172189), 
             'N10': (-0.305679, -0.00571999, -0.322071, -0.17217), 
             'N11': (0.00565427, 0.305554, -0.322243, -0.172254), 
             'N12': (0.31684, 0.616183, -0.322099, -0.172063), 
             'N13': (0.627264, 0.925858, -0.321792, -0.171887), 
             'N14': (-0.926057, -0.62726, -0.485961, -0.336213), 
             'N15': (-0.616498, -0.317089, -0.486444, -0.336606), 
             'N16': (-0.30558, -0.00578257, -0.486753, -0.336864), 
             'N17': (0.00532179, 0.305123, -0.486814, -0.33687), 
             'N18': (0.316662, 0.616018, -0.486495, -0.336537), 
             'N19': (0.62708, 0.92578, -0.485992, -0.336061), 
             'N20': (-0.770814, -0.471826, -0.650617, -0.500679), 
             'N21': (-0.460777, -0.161224, -0.650817, -0.501097), 
             'N22': (-0.149847, 0.149886, -0.650816, -0.501308), 
             'N23': (0.161001, 0.460566, -0.650946, -0.501263), 
             'N24': (0.47163, 0.770632, -0.650495, -0.500592), 
             'N25': (-0.615548, -0.316352, -0.814774, -0.665052), 
             'N26': (-0.305399, -0.00591217, -0.814862, -0.665489), 
             'N27': (0.00550714, 0.304979, -0.815022, -0.665418), 
             'N28': (0.316126, 0.615276, -0.814707, -0.664908), 
             'N29': (-0.46018, -0.16101, -0.97887, -0.829315), 
             'N30': (-0.150043, 0.149464, -0.978648, -0.829007), # APPROXIMATE
             'N31': (0.160884, 0.460147, -0.978775, -0.829426),
             'S1': (-1.08096, -0.782554, 0.00715956, 0.15689), 
             'S2': (-0.7713, -0.47242, 0.0074194, 0.157269), 
             'S3': (-0.4611, -0.161377, 0.00723009, 0.157192), 
             'S4': (-0.149836, 0.150222, 0.00737069, 0.157441), 
             'S5': (0.161297, 0.461031, 0.0072399, 0.1572), 
             'S6': (0.472537, 0.771441, 0.00728934, 0.157137), 
             'S7': (0.782516, 1.08097, 0.00742809, 0.15709), 
             'S8': (-0.92583, -0.627259, 0.171786, 0.32173), 
             'S9': (-0.616329, -0.31694, 0.171889, 0.321823), 
             'S10': (-0.305695, -0.00579187, 0.172216, 0.322179), 
             'S11': (0.00556739, 0.305472, 0.172237, 0.322278), 
             'S12': (0.316973, 0.61631, 0.172015, 0.322057), 
             'S13': (0.627389, 0.925972, 0.171749, 0.321672), 
             'S14': (-0.925847, -0.627123, 0.335898, 0.48578), 
             'S15': (-0.616201, -0.316839, 0.336498, 0.486438), 
             'S16': (-0.305558, -0.00574858, 0.336904, 0.486749), 
             'S17': (0.00557115, 0.305423, 0.33675, 0.486491), 
             'S18': (0.316635, 0.615931, 0.33649, 0.486573), 
             'S19': (0.627207, 0.925969, 0.336118, 0.485923), 
             'S20': (-0.770675, -0.471718, 0.500411, 0.65042), 
             'S21': (-0.46072, -0.161101, 0.501198, 0.650786), 
             'S22': (-0.149915, 0.14982, 0.501334, 0.650856), 
             'S23': (0.160973, 0.460482, 0.501075, 0.650896), 
             'S24': (0.47167, 0.770647, 0.50045, 0.650441), 
             'S25': (-0.615564, -0.316325, 0.66501, 0.814674), 
             'S26': (-0.30512, -0.0056517, 0.665531, 0.81505), 
             'S27': (0.00560886, 0.305082, 0.665509, 0.815022), 
             'S28': (0.316158, 0.615391, 0.665058, 0.814732), 
             'S29': (-0.46021, -0.160988, 0.829248, 0.978699), 
             'S30': (-0.150043, 0.149464, 0.829007, 0.978648), 
             'S31': (0.160898, 0.460111, 0.82932, 0.978804) }

#Correspondence between CCD name and number
ccd_name_to_num =  {
    'S29': 1, 'S30':  2, 'S31':  3, 'S25':  4, 'S26':  5, 'S27':  6, 'S28':  7, 'S20':  8, 'S21':  9, 'S22':  10, 
    'S23': 11, 'S24':  12, 'S14':  13, 'S15':  14, 'S16':  15, 'S17':  16, 'S18':  17, 'S19':  18, 'S8':  19, 'S9':  20, 
    'S10': 21, 'S11':  22, 'S12':  23, 'S13':  24, 'S1' : 25, 'S2':  26, 'S3':  27, 'S4':  28, 'S5':  29, 'S6':  30, 
    'S7':  31, 'N1':  32, 'N2':  33, 'N3':  34, 'N4':  35, 'N5':  36, 'N6':  37, 'N7':  38, 'N8':  39, 'N9':  40, 
    'N10': 41, 'N11':  42, 'N12':  43, 'N13':  44, 'N14':  45, 'N15':  46, 'N16':  47, 'N17':  48, 'N18':  49, 
    'N19': 50, 'N20':  51, 'N21':  52, 'N22':  53, 'N23':  54, 'N24':  55, 'N25':  56, 'N26':  57, 'N27':  58, 'N28':  59, 'N29':  60, 'N30':  61, 'N31':  62
}
ccd_num_to_name = {v: k for k, v in ccd_name_to_num.items()}

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
        return (
            self.ra_00 < ra and
            self.ra_11 > ra and
            self.dec_00 < dec and
            self.dec_11 > dec
        )

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
