import numpy as np

def lonlat_to_abi_scan_angles(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    elevation_m: np.ndarray,
    *,
    satellite_height: float,
    semi_major_axis: float,
    semi_minor_axis: float,
    eccentricity: float,
    longitude_of_projection_origin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert lon/lat/elevation to GOES ABI fixed-grid scan angles.

    Inputs:
      lon_deg, lat_deg: degrees
      elevation_m: meters above ellipsoid
      satellite_height, semi_major_axis, semi_minor_axis: meters

    Returns:
      x, y scan angles in radians.
    """
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    lon_0 = np.radians(longitude_of_projection_origin)

    # Geocentric latitude.
    lat_geo = np.arctan((semi_minor_axis**2 / semi_major_axis**2) * np.tan(lat))

    # Distance from Earth center to ellipsoid surface at geocentric latitude.
    rc = semi_minor_axis / np.sqrt(
        1.0 - eccentricity**2 * np.cos(lat_geo) ** 2
    )

    # Add terrain height. Treat NaN DEM cells as ellipsoid height.
    z = np.nan_to_num(elevation_m, nan=0.0)
    rc = rc + z

    sx = satellite_height - rc * np.cos(lat_geo) * np.cos(lon - lon_0)
    sy = -rc * np.cos(lat_geo) * np.sin(lon - lon_0)
    sz = rc * np.sin(lat_geo)

    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))

    return x, y


