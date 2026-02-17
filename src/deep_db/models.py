from sqlalchemy import Integer, Table, Column, ForeignKey, String, Float, UniqueConstraint, JSON, DateTime, Index
from sqlalchemy.orm import relationship, declarative_base
from astropy_healpix import HEALPix
import astropy.units as u
from astropy.coordinates import SkyCoord

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
             'N30': (-0.150043, 0.149464, -0.829007, -0.978648), # APPROXIMATE
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

Base = declarative_base()

object_orbit_m2m = Table(
    'object_orbit_association',
    Base.metadata,
    Column('object_id', Integer, ForeignKey('solar_system_object.id'), primary_key=True),
    Column('orbit_id', Integer, ForeignKey('orbit.id'), primary_key=True)
)

# Observation model
class Detector(Base):
    __tablename__ = 'detector'
    id = Column(Integer, primary_key=True)
    number = Column(Integer)

class Exposure(Base):
    __tablename__ = 'exposure'
    id = Column(Integer, primary_key=True)
    night_id = Column(Integer, ForeignKey('night.id'), nullable=False, index=True)
    field_id = Column(Integer, ForeignKey('field.id'), nullable=True, index=True)

    expnum = Column(Integer, unique=True)
    caldat = Column(String)
    exposure = Column(Float)
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

    detector = relationship("Detector")
    exposure = relationship("Exposure")

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

class Detection(Base):
    __tablename__ = 'detection'
    id = Column(Integer, primary_key=True)
    detector_exposure_id = Column(Integer, ForeignKey('detector_exposure.id'), nullable=False, index=True)

    ra = Column(Float)
    dec = Column(Float)
    hp_index = Column(Integer, index=True)
    flux = Column(Float)
    flux_err = Column(Float)
    mag = Column(Float)
    mag_err = Column(Float)
    snr = Column(Float)
    zero_point = Column(Float)

    detector_exposure = relationship("DetectorExposure")

class SolarSystemObject(Base):
    __tablename__ = 'solar_system_object'
    id = Column(Integer, primary_key=True)
    binary_properties_id = Column(Integer, ForeignKey('fakes_binary_properties.id'), nullable=True, index=True)
    light_curve_properties_id = Column(Integer, ForeignKey('fakes_light_curve_properties.id'), nullable=True, index=True)

    name = Column(String, unique=True, index=True)
    type = Column(String, index=True)

    orbits = relationship(
        "Orbit", 
        secondary=object_orbit_m2m,
        back_populates="objects"
    )
    ephemerides = relationship("Ephemeris", back_populates="object")
    light_curve_properties = relationship("FakeLightCurveProperties", back_populates="objects", foreign_keys=[light_curve_properties_id])
    binary_properties = relationship("FakeBinaryProperties", back_populates="objects", foreign_keys=[binary_properties_id])

class Orbit(Base):
    __tablename__ = 'orbit'
    id = Column(Integer, primary_key=True)
    epoch = Column(Float, nullable=False)

    objects = relationship(
        "SolarSystemObject", 
        secondary=object_orbit_m2m,
        back_populates="orbits"
    )
    cartesian_state = relationship("CartesianState", back_populates="orbit", uselist=False)
    keplerian_state = relationship("KeplerianState", back_populates="orbit", uselist=False)

class CartesianState(Base):
    __tablename__ = 'cartesian_state'
    id = Column(Integer, primary_key=True)
    orbit_id = Column(Integer, ForeignKey('orbit.id'), nullable=False, unique=True, index=True)

    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    vx = Column(Float)
    vy = Column(Float)
    vz = Column(Float)

    orbit = relationship("Orbit", back_populates="cartesian_state")

class KeplerianState(Base):
    __tablename__ = 'keplerian_state'
    id = Column(Integer, primary_key=True)
    orbit_id = Column(Integer, ForeignKey('orbit.id'), nullable=False, unique=True, index=True)

    a = Column(Float)
    e = Column(Float)
    i = Column(Float)
    Omega = Column(Float)
    omega = Column(Float)
    M = Column(Float, nullable=True)
    Tp = Column(Float, nullable=True)

    orbit = relationship("Orbit", back_populates="keplerian_state")

class MPCTracklet(Base):
    __tablename__ = 'mpc_tracklet'
    id = Column(Integer, primary_key=True)
    trksub = Column(String, index=True)
    trkid = Column(String, index=True)
    object_id = Column(Integer, ForeignKey('solar_system_object.id'), nullable=True, index=True)

    __table_args__ = (
        UniqueConstraint('object_id', 'trksub', 'trkid', name='object_trksub_trkid_uc'),
    )

    object = relationship("SolarSystemObject")

class MPCObservation(Base):
    __tablename__ = 'mpc_observation'
    id = Column(Integer, primary_key=True)
    tracklet_id = Column(Integer, ForeignKey('mpc_tracklet.id'), nullable=False, index=True)
    detector_exposure_id = Column(Integer, ForeignKey('detector_exposure.id'), nullable=True, index=True)

    __table_args__ = (
        UniqueConstraint('tracklet_id', 'detector_exposure_id', name='tracklet_detector_exposure_uc'),
    )

    obstime = Column(DateTime, index=True)
    ra = Column(Float)
    dec = Column(Float)
    hp_index = Column(Integer, index=True)

    mag = Column(Float)
    band = Column(String)
    status = Column(String)

    tracklet = relationship("MPCTracklet")
    detector_exposure = relationship("DetectorExposure")

class FakeLightCurveProperties(Base):
    __tablename__ = 'fakes_light_curve_properties'
    id = Column(Integer, primary_key=True)

    h_vr = Column(Float)
    amp = Column(Float)
    period = Column(Float)
    phase = Column(Float)

    objects = relationship("SolarSystemObject", back_populates="light_curve_properties")

class FakeBinaryProperties(Base):
    __tablename__ = 'fakes_binary_properties'
    id = Column(Integer, primary_key=True)

    delta_h = Column(Float)
    separation = Column(Float)
    angle = Column(Float)

    objects = relationship("SolarSystemObject", back_populates="binary_properties")

class EphemerisSource(Base):
    __tablename__ = 'ephemeris_source'
    id = Column(Integer, primary_key=True)

    name = Column(String, unique=True, index=True) # e.g. JPL / SkyBot / MPC / orbit / manual

class Ephemeris(Base):
    __tablename__ = 'ephemeris'
    id = Column(Integer, primary_key=True)

    object_id = Column(Integer, ForeignKey('solar_system_object.id'), nullable=False)
    source_id = Column(Integer, ForeignKey('ephemeris_source.id'), nullable=False, index=True)
    detector_exposure_id = Column(Integer, ForeignKey('detector_exposure.id'), nullable=True, index=True)

    __table_args__ = (
        UniqueConstraint('object_id', 'source_id', 'detector_exposure_id', name='object_source_detector_exposure_uc'),
    )

    ra = Column(Float)
    dec = Column(Float)
    hp_index = Column(Integer, index=True)
    ra_rate = Column(Float)
    dec_rate = Column(Float)
    v_mag = Column(Float)
    alpha = Column(Float)
    delta = Column(Float)
    r = Column(Float)

    object = relationship("SolarSystemObject", back_populates="ephemerides")
    source = relationship("EphemerisSource")
    detector_exposure = relationship("DetectorExposure")

class Repository(Base):
    __tablename__ = 'repository'
    id = Column(Integer, primary_key=True)

    name = Column(String)
    host = Column(String)
    path = Column(String)

    __table_args__ = (
        UniqueConstraint('name', 'host', 'path', name='repo_uc'),
    )

class EphemerisDetectorLocation(Base):
    __tablename__ = 'ephemeris_detector_location'
    id = Column(Integer, primary_key=True)
    ephemeris_id = Column(Integer, ForeignKey('ephemeris.id'), nullable=False, index=True)
    repository_id = Column(Integer, ForeignKey('repository.id'), nullable=False, index=True)

    x = Column(Float)
    y = Column(Float)
    collection = Column(String, index=True)
    dataset = Column(String, index=True)

    __table_args__ = (
        UniqueConstraint('ephemeris_id', 'repository_id', 'collection', 'dataset', name='ephemeris_repo_collection_dataset_uc'),
        Index('idx_ephemeris_detector_location_composite', 'ephemeris_id', 'repository_id', 'collection', 'dataset'),
    )

    ephemeris = relationship("Ephemeris")

class Cutout(Base):
    __tablename__ = 'cutout'
    id = Column(Integer, primary_key=True)
    ephemeris_detector_location_id = Column(
        Integer, ForeignKey('ephemeris_detector_location.id'), 
        nullable=False, unique=True, index=True
    )
    
    image = Column(JSON, nullable=True)
    variance = Column(JSON, nullable=True)
    mask = Column(JSON, nullable=True)

    path = Column(String, nullable=True)
    width = Column(Integer)
    height = Column(Integer)
    
    ephemeris_detector_location = relationship("EphemerisDetectorLocation")

class Photometry(Base):
    __tablename__ = "photometry"
    id = Column(Integer, primary_key=True)
    ephemeris_detector_location_id = Column(
        Integer, ForeignKey('ephemeris_detector_location.id'),
        nullable=False, unique=True, index=True
    )

    logl = Column(Float)
    a = Column(Float)
    c = Column(Float)
    flux = Column(Float)
    flux_ref = Column(Float)
    sigma = Column(Float)
    sigma_ref = Column(Float)
    snr = Column(Float)
    mag = Column(Float)
    mag_err_lo = Column(Float)
    mag_err_hi = Column(Float)
    mask = Column(Integer)
    zero_point = Column(Float)

    ephemeris_detector_location = relationship("EphemerisDetectorLocation")
