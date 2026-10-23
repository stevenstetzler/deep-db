import re

_MPCHECKER_UNPACKED_NAME_REGEX = re.compile(
    r"(?:(?P<number>\(\d+\))\s?)?(?:(?P<provid>[0-9]{4}\s[A-Z0-9\-]+)|(?P<name>[A-Za-z]+))"
)

def _parse_mpchecker_designation(unpacked_name):
    m = _MPCHECKER_UNPACKED_NAME_REGEX.match(unpacked_name)
    if not m:
        raise Exception(f"{unpacked_name} doesn't match pattern")
    d = m.groupdict()
    permanent_designation = d['number']
    if permanent_designation:
        permanent_designation = permanent_designation.replace("(", "").replace(")", "")
    # (unpacked_provisional_designation, permanent_designation, name)
    return d['provid'], permanent_designation, d['name']

def fetch_rows_for_exposures(exposures):
    from astropy.units import deg
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from jorbit.mpchecker import mpchecker

    for exposure in exposures:
        objects = mpchecker(
            SkyCoord(exposure.ra * deg, exposure.dec * deg),
            Time(exposure.midpoint, scale='utc'),
            radius=1.2 * deg,
            observer="cerro tololo observatory, la serena",
            extra_precision=True,
        )
        designations = [_parse_mpchecker_designation(obj['Unpacked Name']) for obj in objects]
        for obj, designation in zip(objects, designations):
            matches = list(filter(lambda de: de.contains(obj['ra'] * deg, obj['dec'] * deg), exposure.detector_exposures))
            detnums = [int(m.detector.number) for m in matches] if matches else [None]
            for detnum in detnums:
                yield dict(
                    provid=designation[0],
                    permid=designation[1],
                    name=designation[2],
                    expnum=exposure.expnum,
                    detnum=detnum,
                    ra=float(obj['ra']),
                    dec=float(obj['dec']),
                    v_mag=float(obj['est. Vmag']),
                )


def fetch_mpchecker(mpc_db, exposures, echo=False, output=None, format='ascii.ecsv'):
    from astropy.table import Table
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url
    from sqlalchemy.orm import Session
    from sys import stdout
    from sqlalchemy import text
    import numpy as np

    _output = stdout
    if output is not None:
        _output = open(output, 'w')

    table = Table(list(fetch_rows_for_exposures(exposures)))
    missing_provid = table['provid'] == None
    permid_to_get = table[missing_provid]['permid']
    if len(permid_to_get) > 0:
        permid_to_get = ",".join(map(lambda x : f"'{x}'", permid_to_get))

        # some minor planet names (e.g. "Strömgren") are non-ASCII; without
        # this, psycopg2 can negotiate an ascii client_encoding (depending on
        # the server/OS locale) and blow up decoding those rows. Only passed
        # for postgres -- it isn't a recognized create_engine() argument for
        # other dialects (e.g. the sqlite engines used in tests).
        mpc_engine_kwargs = {}
        if make_url(mpc_db).get_backend_name() == 'postgresql':
            mpc_engine_kwargs['client_encoding'] = 'utf8'

        permid_lookup = {}
        engine = create_engine(mpc_db, **mpc_engine_kwargs)
        with Session(engine) as session:
            q = text(f"""
                SELECT
                    unpacked_primary_provisional_designation AS provid,
                    permid
                FROM numbered_identifications
                WHERE permid IN ({permid_to_get})
            """)
            for row in session.execute(q).mappings():
                permid_lookup[row['permid']] = row['provid']

        # np.where(missing_provid) returns a 1-tuple holding the whole
        # index array -- iterating it directly binds idx to that entire
        # array (one loop iteration total), so `table[idx]['permid'][0]`
        # only ever read the *first* missing-provid row's permid and then
        # stamped its resolved provid onto every missing-provid row in the
        # table. Unwrap with [0] to get the actual index array and loop
        # per-row instead.
        for idx in np.where(missing_provid)[0]:
            permid = table['permid'][idx]
            table['provid'][idx] = permid_lookup[permid]

    table.write(_output, format=format)
    
    if _output is not stdout:
        _output.close()