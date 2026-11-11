import numpy as np
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
from astropy.units import deg

hp = HEALPix(nside=256, order='nested', frame='icrs')

def ra_dec_to_coordinate(ra, dec):
    hp_index = hp.lonlat_to_healpix(ra, dec)
    coord = SkyCoord(ra=ra, dec=dec, frame='icrs')
    return float(coord.ra.to(deg).value), float(coord.dec.to(deg).value), int(hp_index)

def sky_to_gnomonic(ra, dec, ra0, dec0):
    """
    Forward gnomonic (tangent plane) projection of (ra, dec) about tangent
    point (ra0, dec0). All angles in degrees.

    Returns the standard tangent-plane coordinates (xi, eta) in degrees,
    where xi ~ (ra - ra0) * cos(dec0) and eta ~ dec - dec0 for small offsets
    from the tangent point -- the convention DESTNOSIM's ccdBounds are in.
    """
    ra_r, dec_r, ra0_r, dec0_r = np.radians([ra, dec, ra0, dec0])
    dra = ra_r - ra0_r
    denom = np.sin(dec0_r) * np.sin(dec_r) + np.cos(dec0_r) * np.cos(dec_r) * np.cos(dra)
    xi = np.cos(dec_r) * np.sin(dra) / denom
    eta = (np.cos(dec0_r) * np.sin(dec_r) - np.sin(dec0_r) * np.cos(dec_r) * np.cos(dra)) / denom
    return np.degrees(xi), np.degrees(eta)

def gnomonic_to_sky(xi, eta, ra0, dec0):
    """
    Inverse of sky_to_gnomonic(): given tangent-plane offsets (xi, eta) in
    degrees about tangent point (ra0, dec0) in degrees, return (ra, dec) in
    degrees. ra is wrapped to [0, 360).
    """
    xi_r, eta_r, ra0_r, dec0_r = np.radians([xi, eta, ra0, dec0])
    denom = np.cos(dec0_r) - eta_r * np.sin(dec0_r)
    ra = ra0_r + np.arctan2(xi_r, denom)
    dec = np.arctan2(np.sin(dec0_r) + eta_r * np.cos(dec0_r), np.hypot(denom, xi_r))
    return np.degrees(ra) % 360.0, np.degrees(dec)
