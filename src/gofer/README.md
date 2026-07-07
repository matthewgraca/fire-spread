# The GOFER algorithm
Paper: https://essd.copernicus.org/articles/16/1395/2024/

At a high level, the GOFER algorithm is simply a way to draw fire perimeters using GOES satellite data.

![](https://essd.copernicus.org/articles/16/1395/2024/essd-16-1395-2024-avatar-web.png)

# Remapping
As a part of the GOFER algorithm pulled from [**Google's Restif and Hoffman**](https://medium.com/google-earth/how-to-generate-wildfire-boundary-maps-with-earth-engine-b38eadc97a38), fire mask values are converted to confidence values. These confidence values effectively parameterize the mask, introducing acceptable/unacceptable thresholds for determining if a fire pixel should be included/excluded on the basis of criteria like quality and agreement between the two satellites on if fire occupies the same pixel.

# Ingest
## GOES AWS
Ingestion pipeline uses [**goes2go**](https://goes2go.readthedocs.io/en/latest/) by Brian Blaylock. Not the fastest thing in the world, but it's proven to be robust and gets the job done as a downloader for GOES data.

The product we pull from the GOES AWS bucket is the [**Fire Detection and Characterization product**](https://www.goes-r.gov/education/docs/fs_fire.pdf). This data comes in four flavors:
- Mask
- Temperature
- Fire Radiative Power
- Area

Our goal is to draw fire perimeters and track fire spread. To that end, we use "Mask", which is just a binary value for if there is a fire detected on a given pixel. Other variables give insight into the detected strength of the fire at that pixel.

![](https://www.star.nesdis.noaa.gov/goesr/images/internal/Kenneth_2025009231117_w_big_legend.png)

### Readings
Another background source: https://www.star.nesdis.noaa.gov/goesr/product_land_fire.php

## CalFire/FRAP
The pipeline also includes using CalFire/FRAP's [**California Fire Perimeters dataset**](https://data.ca.gov/dataset/california-fire-perimeters-1950). This includes crucial information like fire start/end, precise lat/lon of the fire, dates, and the full polygon of the final fire perimeter. This data (particularly the dates) are used to grab the relevant data.

![](https://www.esri.com/arcgis-blog/app/uploads/2024/10/Application-Fires-Burned.png)

### Readings
Play around with perimeter data on CalFire's Esri map: https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters

# Temporal Downsampler
While the GOFER product is hourly, it makes use of the fact that GOES produces observations every 5 minutes (for 12 observations per hour). GOFER leverages this extra data by grabbing all the subhourly observations, remapping the pixels to confidence values, then grabbing the max confidence pixels.

For example, for the hour 3:00, observations from (2:00, 3:00] are gathered. The observations are then compressed into one by max confidence. This significantly reduces the variance of our observations due to uncertainty (at least for the hour). After these observations are aggregated, they are assigned to the timestep 3:00.

On top of this aggregation, this set of methods is also responsible for cleaning and alignment of timesteps. Any corrupt timesteps has their observations removed. This is usually due to erroneous storage on GOES's end. Then, any missing timesteps are automatically given observations with 0 confidence. 


# Orthorectification
## Overview
Due to off-nadir scan angles of geostationary satellites and complex terrain, imagery is subject to a parallax effect. Satellite images should undergo orthorectification to improve correctness.

Super interesting stuff, but I'm definitely not qualified to speak more on the matter.

![](https://raw.githubusercontent.com/spestana/goes-ortho/main/docs/images/GOES-terrain-correction.gif)

## Creating the orthographic map
With regards to the creation of the orthographic map, we do two things of note:
1. Create fixed grid diagnostics
2. Perform parallax adjustment with a given factor

### Fixed grid diagnostics
Fixed grid diagnostics refer to which original ABI pixels the DEM-grid sampled from. Recall that DEM is sub-pixel (30m) from the perspective of GOES (2km); so we have to make a choice on what pixel on the DEM-grid will take from GOES, where we use nearest-neighbor.

`abi_fixed_grid_x`, `abi_fixed_grid_y`, and `zone_labels` records the GOES grid cells selected.

This tells you things like how many of the same GOES pixel was used for a given bundle of pixels on the DEM-grid.

### Parallax Adjustment Factor
Regarding parallax adjustment factor; GOFER authors experimentally identified that an adjustment factor is useful for the fire mask product, instead of using the full correction.

Why not use orthorectification as-is (like in Spetsana)? Because full orthorectification is correct for a known surface point; but GOFER’s active-fire pixels are not known surface points; they are coarse, mixed, processed fire-confidence signals used to build perimeters. The adjustment factor lets the algorithm apply the physically expected correction while damping overcorrection caused by sub-pixel fire location uncertainty, DEM/pixel scale mismatch, smoothing, thresholding, and sensor/view-angle biases.

At the end of the day, the authors observe higher IOU with fire perimeters when the parallax adjusment is slightly dampened with the GOFER-Combined product.

## Sources and Further Reading
Orthorectification logic adapted from Spetsana. Includes a good explanation as well!
- Repo: https://github.com/spestana/goes-ortho

Coordinate transformation (or geolocation) between GOES ABI scan angles and lat/lon
- Practical: https://lsterzinger.medium.com/add-lat-lon-coordinates-to-goes-16-goes-17-l2-data-and-plot-with-cartopy-27f07879157f
- Theory and Math: https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm

# Compositing
This one is quite simple; it's basically combining two different satellite images onto one coherent map. Now, this would normally be a pain because you'd have to move each satellite to lat/lon, then painfully align both lat/lon grids to have the same elements... but thankfully, our orthorectification geolocates each satellite onto the EXACT same grid defined by our elevation map! So it's a simple merge on whatever data variable; we use the average here.

# Spatial smoothing
The goal of this is to smooth out the jagged edges of the perimeters. We do this using a neighborhood mean convovling over the image. For example, a 3x3 kernel means that the center of the box takes on the mean of the other 8 pixels; done for every pixel. The paper recommends using a dynamic size kernel for this smoothing, as opposed to Google which uses a static 2km kernel.

The authors determined that these kernel sizes worked best for their large California fires: 
- GOES-East: 3.1–3.6 km
- GOES-West: 2.5–2.7 km
- GOES-Combined: 1.6–1.7 km
