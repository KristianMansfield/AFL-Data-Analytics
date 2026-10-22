#!/usr/bin/env python3

# Where venues and teams are, for distance-based analysis.
#
# The database has no venue locations (venues.city and venues.state are empty) and no team home
# cities, so they're listed here by hand. Locations are city-level: every Melbourne ground is
# "Melbourne", as travel across town is negligible next to interstate trips. Distances are
# straight lines between city centres (great-circle), not travel routes.
#
# Covers every venue used for AFL/VFL games since 1990. A venue missing from VENUES raises an
# error naming it, so add it here when a new ground is used.

import numpy as np

# City -> (latitude, longitude)
CITIES = {
    "Melbourne": (-37.82, 144.98), "Geelong": (-38.16, 144.35), "Ballarat": (-37.54, 143.85),
    "Adelaide": (-34.92, 138.60), "Barossa": (-34.60, 138.89), "Mount Barker": (-35.07, 138.86),
    "Hobart": (-42.88, 147.33), "Launceston": (-41.43, 147.14), "Sydney": (-33.87, 151.10),
    "Canberra": (-35.32, 149.13), "Brisbane": (-27.49, 153.04), "Gold Coast": (-28.00, 153.37),
    "Perth": (-31.95, 115.86), "Bunbury": (-33.33, 115.64), "Darwin": (-12.40, 130.88),
    "Alice Springs": (-23.70, 133.88), "Cairns": (-16.93, 145.75), "Townsville": (-19.32, 146.73),
    "Shanghai": (31.30, 121.51), "Wellington": (-41.27, 174.79),
}

# Venue name (venues.name) -> city
VENUES = {
    "M.C.G.": "Melbourne", "Docklands": "Melbourne", "Princes Park": "Melbourne",
    "Waverley Park": "Melbourne", "Victoria Park": "Melbourne", "Western Oval": "Melbourne",
    "Windy Hill": "Melbourne", "Moorabbin Oval": "Melbourne",
    "Kardinia Park": "Geelong", "Eureka Stadium": "Ballarat",
    "Adelaide Oval": "Adelaide", "Football Park": "Adelaide", "Norwood Oval": "Adelaide",
    "Barossa Park": "Barossa", "Summit Sports Park": "Mount Barker",
    "Bellerive Oval": "Hobart", "North Hobart": "Hobart", "York Park": "Launceston",
    "S.C.G.": "Sydney", "Stadium Australia": "Sydney", "Sydney Showground": "Sydney",
    "Blacktown": "Sydney", "Bruce Stadium": "Canberra", "Manuka Oval": "Canberra",
    "Gabba": "Brisbane", "Carrara": "Gold Coast",
    "Subiaco": "Perth", "W.A.C.A.": "Perth", "Perth Stadium": "Perth", "Hands Oval": "Bunbury",
    "Marrara Oval": "Darwin", "Traeger Park": "Alice Springs", "Cazaly's Stadium": "Cairns",
    "Riverway Stadium": "Townsville", "Jiangwan Stadium": "Shanghai", "Wellington": "Wellington",
}

# Team location (teams.location) -> home city, where it isn't Melbourne. South Melbourne is the
# Sydney Swans (the database keeps the original name), Footscray the Western Bulldogs.
TEAM_CITIES = {
    "Geelong": "Geelong",
    "South Melbourne": "Sydney", "Greater Western Sydney": "Sydney",
    "West Coast": "Perth", "Fremantle": "Perth",
    "Adelaide": "Adelaide", "Port Adelaide": "Adelaide",
    "Brisbane Lions": "Brisbane", "Gold Coast": "Gold Coast",
}


def teamCity(team, year):
    """A team's home city in a season, by teams.location."""
    if team == "Brisbane Bears":
        return "Gold Coast" if year <= 1992 else "Brisbane"  # moved from Carrara to the Gabba in 1993
    if team == "South Melbourne" and year < 1982:
        return "Melbourne"  # moved to Sydney in 1982
    return TEAM_CITIES.get(team, "Melbourne")


def venueCity(venue):
    if venue not in VENUES:
        raise KeyError(f"No location for venue {venue!r}: add it to VENUES in locations.py")
    return VENUES[venue]


def distanceKm(city_a, city_b):
    """Great-circle distance between two cities in kilometres."""
    (lat1, lon1), (lat2, lon2) = np.radians(CITIES[city_a]), np.radians(CITIES[city_b])
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return float(6371 * 2 * np.arcsin(np.sqrt(h)))


def travelKm(team, year, venue):
    """How far a team travelled from its home city to a venue."""
    return distanceKm(teamCity(team, year), venueCity(venue))
