#!/bin/env python3
"""Kise"""
from matplotlib import colors
import support

if __name__ == "__main__":
    #DEM_FILE = "/lustre/storeB/project/nwp/H2O/wp4/TOPD/Pyflwdir/6501_50m_33.tif"
    DEM_FILE ="/lustre/storeA/users/josteinbl/H2O/wp4/TOPD/ISBA-TOPD-coupling/dtm50_6702_50m_33.tif"
    MIN_AREA = 2000


    # TODO merge tif files  # noqa
    # read elevation data of the basin using rasterio
    dem = support.DEMFromTIF(DEM_FILE)
    dem.plot()

    flw_obj = support.pyflwdir_from_dem(dem)
    hillshade = colors.LightSource(azdeg=115, altdeg=45).hillshade(dem.elevtn, vert_exag=1e3)
    support.plot_streams(flw_obj, hillshade, dem,
                         title="Streams based steepest gradient algorithm KISE",
                         file="flw_streams_steepest_gradient_kise.png")

    slope = support.get_slope(dem)
    support.plot_slope(slope, hillshade, dem, title=None, file=None)

    flw_obj = support.pyflwdir_from_dem(dem)
