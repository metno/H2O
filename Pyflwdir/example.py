#!/bin/env python3
"""Select a list of catchments in a DEM"""
from matplotlib import colors
import support


if __name__ == "__main__":

    DTM_AREAS = "/lustre/storeB/project/nwp/H2O/wp4/TOPD/Pyflwdir/NHM50_kartblad.shp"
    ALL_CATCHMENTS = "/lustre/storeB/project/nwp/H2O/wp4/RRdata/shapefiles/utm33shp/" \
                        "NVEData/Nedborfelt/Nedborfelt_Vassdragsomr.shp"
    subs = ["017"]
    DEM_FILE = "/lustre/storeB/project/nwp/H2O/wp4/TOPD/Pyflwdir/6501_50m_33.tif"
    MIN_AREA = 2000

    hdatametasub = support.get_dtm_areas(DTM_AREAS)
    regs2sub = support.regs2subcatchments(ALL_CATCHMENTS)

    support.plot_dtm_info(regs2sub, hdatametasub, title="DTM overview with catchments")

    # Select sub catchments
    regs2sub = support.select_catchments(regs2sub, subs)

    # TODO merge tif files  # noqa
    # read elevation data of the basin using rasterio
    # masked_dem = support.DEMFromTIF(DEM_FILE)
    masked_dem = support.MaskedDEMFromTIF(DEM_FILE, regs2sub)

    x0 = 962500
    y0 = 3560000
    geo = support.Geo("l2e", x0, y0, 1500, 1200, 50)

    # geo.plot_map()
    tmap = support.TopdMapFromGeo(geo, masked_dem)
    tmap.plot_map()
    tmap.save_to_file("test.map")

    tmap = support.TopdMapFromFile("test.map")
    tmap.plot_map()

    hillshade = colors.LightSource(azdeg=115, altdeg=45).hillshade(masked_dem.elevtn, vert_exag=1e3)    
    flw_obj = support.pyflwdir_from_dem(masked_dem)
    hillshade = colors.LightSource(azdeg=115, altdeg=45).hillshade(masked_dem.elevtn, vert_exag=1e3)
    distance = support.stream_distance(flw_obj, masked_dem, unit='cell')
    support.plot_stream_distance(distance, masked_dem, hillshade, title=None, file=None)

    slope = support.get_slope(masked_dem)
    support.plot_slope(slope, hillshade, masked_dem, title=None, file=None)

    support.plot_streams(flw_obj, hillshade, masked_dem,
                 title=f"Streams based steepest gradient algorithm {str(subs)}",
                 file="flw_streams_steepest_gradient_" + "".join(subs) + ".png"
    )

    support.write_d8_tif_file(flw_obj, masked_dem, "".join(subs) + "_flwdir_d8.tif")

    gdf_subbas = support.calc_sub_basin(flw_obj, masked_dem, min_area=MIN_AREA)
    support.plot_sub_basin(gdf_subbas, masked_dem,
                   title=f"Subbasins based on a minimum area of {MIN_AREA} km2 {str(subs)}",
                   file="flw_streams_subbasins_" + "".join(subs) + ".png")
