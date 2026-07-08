# Database Schema

This repository defines its database schema in `src/deep_db/models.py` using SQLAlchemy declarative models.

## Core observation metadata
- `night`: observing nights (`night`, `start_time`, `end_time`)
- `field`: field identifiers (`name`)
- `detector`: detector IDs/numbers
- `exposure`: exposure-level metadata (timing, pointing, band, target), linked to `night` and `field`
- `detector_exposure`: detector-level geometry per exposure (center/corner coordinates, HEALPix indices), linked to `detector` and `exposure`
- `detection`: detection photometry linked to `detector_exposure`

## Object and orbit model
- `solar_system_object`: object identity/type and optional links to fake-property tables
- `orbit`: orbit epoch container
- `object_orbit_association`: many-to-many join between objects and orbits
- `cartesian_state`: Cartesian orbit state (`x,y,z,vx,vy,vz`) linked 1:1 to `orbit`
- `keplerian_state`: Keplerian elements (`a,e,i,Omega,omega,M,Tp`) linked 1:1 to `orbit`

## Ephemerides and sources
- `ephemeris_source`: ephemeris provenance (e.g., JPL, MPC)
- `ephemeris`: per-object positional predictions and rates, linked to object/source and optionally to detector exposure

## MPC-linked tables
- `mpc_tracklet`: MPC tracklet identity, optionally linked to object
- `mpc_observation`: MPC observation points linked to tracklets and optionally detector exposure

## Simulated/fake object properties
- `fakes_light_curve_properties`: variability model fields (`h_vr`, `amp`, `period`, `phase`)
- `fakes_binary_properties`: binary parameters (`delta_h`, `separation`, `angle`)

## Repository, location, and derived products
- `repository`: repository identity (`name`, `host`, `path`)
- `ephemeris_detector_location`: object location (`x`, `y`) for a repository/collection/dataset and ephemeris
- `cutout`: cutout artifact metadata (JSON payloads or file `path`, dimensions) linked 1:1 to `ephemeris_detector_location`
- `photometry`: derived forced-photometry outputs linked 1:1 to `ephemeris_detector_location`

## Relationship overview
- `night` 1:N `exposure`
- `field` 1:N `exposure`
- `detector` N:M `exposure` through `detector_exposure`
- `solar_system_object` N:M `orbit` through `object_orbit_association`
- `solar_system_object` 1:N `ephemeris`
- `ephemeris_source` 1:N `ephemeris`
- `ephemeris` 1:N `ephemeris_detector_location`
- `ephemeris_detector_location` 1:1 `cutout`
- `ephemeris_detector_location` 1:1 `photometry`
