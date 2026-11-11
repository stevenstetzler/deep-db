import numpy as np
import pytest

from astropy.units import deg

from deep_db.database.survey import Detector, DetectorExposure, Exposure
from deep_db.util.coord import gnomonic_to_sky, sky_to_gnomonic
from deep_db.util.decam import ccdBounds, ccd_name_to_num, ccd_num_to_name

BORESIGHTS = [(150.0, 10.0), (200.0, -25.0), (359.8, 5.0), (0.3, -12.0), (10.0, 30.0)]


def make_de(name, ra0, dec0):
    # transient ORM objects (no DB session needed)
    return DetectorExposure(
        exposure=Exposure(ra=ra0, dec=dec0),
        detector=Detector(number=ccd_name_to_num[name]),
    )


def contains(de, ra, dec):
    # callers in fetch/mpc.py pass astropy Quantities
    return de.contains(ra * deg, dec * deg)


def test_ccd_bounds_ordered_and_numbering_consistent():
    for name, (xmin, xmax, ymin, ymax) in ccdBounds.items():
        assert xmin < xmax and ymin < ymax, name
    assert len(ccdBounds) == 62
    assert all(ccd_num_to_name[ccd_name_to_num[n]] == n for n in ccdBounds)


@pytest.mark.parametrize("ra0,dec0", BORESIGHTS)
def test_roundtrip(ra0, dec0):
    for xi in np.linspace(-1.1, 1.1, 9):
        for eta in np.linspace(-1.0, 1.0, 9):
            ra, dec = gnomonic_to_sky(xi, eta, ra0, dec0)
            xi2, eta2 = sky_to_gnomonic(ra, dec, ra0, dec0)
            assert xi2 == pytest.approx(xi, abs=1e-10)
            assert eta2 == pytest.approx(eta, abs=1e-10)
            assert 0 <= ra < 360


def test_boresight_maps_to_origin():
    assert sky_to_gnomonic(150.0, 10.0, 150.0, 10.0) == (0.0, 0.0)


@pytest.mark.parametrize("ra0,dec0", BORESIGHTS)
@pytest.mark.parametrize("name", list(ccdBounds))
def test_contains_center_and_inset_corners(name, ra0, dec0):
    xmin, xmax, ymin, ymax = ccdBounds[name]
    de = make_de(name, ra0, dec0)
    ra, dec = gnomonic_to_sky((xmin + xmax) / 2, (ymin + ymax) / 2, ra0, dec0)
    assert contains(de, ra, dec)
    # just inside / just outside each edge
    eps = 1e-3
    for dx, dy, inside in [(eps, eps, True), (-eps, -eps, True)]:
        x = xmin + eps if dx > 0 else xmax - eps
        y = ymin + eps if dy > 0 else ymax - eps
        assert contains(de, *gnomonic_to_sky(x, y, ra0, dec0)) == inside
    assert not contains(de, *gnomonic_to_sky(xmin - eps, (ymin + ymax) / 2, ra0, dec0))
    assert not contains(de, *gnomonic_to_sky((xmin + xmax) / 2, ymax + eps, ra0, dec0))


def test_each_point_in_exactly_one_chip_or_gap():
    ra0, dec0 = 359.9, 8.0  # straddles RA=0
    rng = np.random.default_rng(0)
    des = [make_de(n, ra0, dec0) for n in ccdBounds]
    for xi, eta in rng.uniform([-1.0, -1.0], [1.0, 1.0], size=(300, 2)):
        ra, dec = gnomonic_to_sky(xi, eta, ra0, dec0)
        hits = [d for d in des if contains(d, ra, dec)]
        assert len(hits) <= 1


def test_flat_offsets_were_wrong_at_dec30():
    """Regression for issue #5: flat RA offset misplaces outer chip by >> 100 arcsec."""
    ra0, dec0 = 100.0, 30.0
    xmin, xmax, ymin, ymax = ccdBounds["N7"]
    ra, dec = gnomonic_to_sky(xmax, ymax, ra0, dec0)
    flat_ra = ra0 + xmax
    assert abs(ra - flat_ra) * 3600 > 100
    assert contains(make_de("N7", ra0, dec0), *gnomonic_to_sky(xmax - 0.01, ymax - 0.01, ra0, dec0))
