def fetch_rows_for_exposures(mpc_session, exposures):
    """
    query the mpc_sbn joined tables to get the rows for the provided exposures
    select over all exposures provided and then return rows grouped by exposure_id
    rows should have 
    """
    from astropy.units import deg
    from sqlalchemy import text
    from astropy.time import Time

    for exposure in exposures:
        start_time = exposure.obstime
        end_time = exposure.endtime

        q = text(rf"""
            SELECT
                o.obsid,
                o.trkid,
                o.trksub,
                i.unpacked_primary_provisional_designation as provid,
                n.permid as permid,
                mn.name,
                o.obstime,
                o.ra,
                o.dec,
                o.mag,
                o.status
            FROM obs_sbn o
            LEFT OUTER JOIN current_identifications i -- o.provid may be null
                ON o.provid = i.unpacked_secondary_provisional_designation
            LEFT OUTER JOIN numbered_identifications n -- o.permid may be null
                ON i.packed_primary_provisional_designation = n.packed_primary_provisional_designation
            LEFT OUTER JOIN minor_planet_names mn -- get name if available
                ON n.permid = mn.mp_number
            WHERE
                o.obstime >= '{start_time}' AND
                o.obstime <= '{end_time}' AND
                o.stn = 'W84'
        """)
        for row in mpc_session.execute(q).mappings():
            expnum = None
            if exposure.contains(row['ra'] * deg, row['dec'] * deg):
                expnum = exposure.expnum
            
            detnum = None                
            if expnum is not None:
                matched = list(filter(lambda de : de.contains(row['ra'] * deg, row['dec'] * deg), exposure.detector_exposures))
                if matched:
                    detnum = matched[0].detector.number
            d = {
                "expnum": expnum,
                "detnum": detnum,
                **row
            }
            d['ra'] = float(d['ra'])
            d['dec'] = float(d['dec'])
            d['mag'] = float(d['mag']) if d['mag'] else None          
            d['obstime'] = Time(d['obstime'], scale='utc')
            yield d

def fetch_mpc(mpc_db, exposures, echo=False, output=None, format='ascii.ecsv'):
    from deep_db.util.coord import ra_dec_to_coordinate
    from astropy.time import Time
    from astropy.units import deg
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url
    from sqlalchemy.orm import Session
    from astropy.table import Table
    from sys import stdout

    _output = stdout
    if output is not None:
        _output = open(output, 'w')

    # some minor planet names (e.g. "Strömgren") are non-ASCII; without
    # this, psycopg2 can negotiate an ascii client_encoding (depending on
    # the server/OS locale) and blow up decoding those rows. Only passed
    # for postgres -- it isn't a recognized create_engine() argument for
    # other dialects (e.g. the sqlite engines used in tests).
    mpc_engine_kwargs = {}
    if make_url(mpc_db).get_backend_name() == 'postgresql':
        mpc_engine_kwargs['client_encoding'] = 'utf8'

    engine = create_engine(mpc_db, echo=echo, **mpc_engine_kwargs)
    with Session(engine) as session:
        table = Table(list(fetch_rows_for_exposures(session, exposures)))
        table.write(output, format=format)
    
    if _output is not stdout:
        _output.close()
