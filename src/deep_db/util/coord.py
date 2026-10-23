from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
from astropy.units import deg

hp = HEALPix(nside=256, order='nested', frame='icrs')

def ra_dec_to_coordinate(ra, dec):
    hp_index = hp.lonlat_to_healpix(ra, dec)
    coord = SkyCoord(ra=ra, dec=dec, frame='icrs')
    return float(coord.ra.to(deg).value), float(coord.dec.to(deg).value), int(hp_index)
