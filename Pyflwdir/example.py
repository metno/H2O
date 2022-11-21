#!/bin/env python3
"""Select a list of catchments in a DEM"""
from matplotlib import colors
import support


if __name__ == "__main__":

    DTM_AREAS = "/lustre/storeB/project/nwp/H2O/wp4/TOPD/Pyflwdir/NHM50_kartblad.shp"
    # ALL_CATCHMENTS = "..//Nedborfelt/Nedborfelt_Vassdragsomr.shp"
    ALL_CATCHMENTS = "/lustre/storeB/project/nwp/H2O/wp4/RRdata/shapefiles/utm33shp/" \
                     "NVEData/Nedborfelt/Nedborfelt_Vassdragsomr.shp"
    subs = ["017", "018"]
    #subs = ["017"]
    DEM_FILE = "/lustre/storeB/project/nwp/H2O/wp4/TOPD/Pyflwdir/6501_50m_33.tif"
    MIN_AREA = 2000

    hdatametasub = support.get_dtm_areas(DTM_AREAS)
    regs2sub = support.regs2subcatchments(ALL_CATCHMENTS)

    support.plot_dtm_info(regs2sub, hdatametasub, title="DTM overview with catchments")

    # Select sub catchments
    regs2sub = support.select_catchments(regs2sub, subs)

    # TODO merge tif files  # noqa
    # read elevation data of the basin using rasterio
    extent, topomasked, out_image, nodata, transform, latlon,\
        crs, prof = support.read_elevation_data(regs2sub, DEM_FILE)
    support.plot_dem(topomasked, extent, title=f"Elevation data in {DEM_FILE}")

    flw_obj = support.pyflwdir_from_dem(out_image, nodata, transform, latlon)

    hillshade = colors.LightSource(azdeg=115, altdeg=45).hillshade(topomasked, vert_exag=1e3)
    support.plot_streams(flw_obj, hillshade, extent, crs,
                 title=f"Streams based steepest gradient algorithm {str(subs)}",
                 file="flw_streams_steepest_gradient_" + "".join(subs) + ".png"
    )

    support.write_tif_file(flw_obj, prof, "".join(subs) + "_flwdird.tif")

    gdf_subbas = support.calc_sub_basin(flw_obj, crs, min_area=MIN_AREA)
    support.plot_sub_basin(gdf_subbas,
                   title=f"Subbasins based on a minimum area of {MIN_AREA} km2 {str(subs)}",
                   file="flw_streams_subbasins_" + "".join(subs) + ".png")
