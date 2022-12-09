#!/bin/env python3
"""Kise"""
import support
import logging


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    DEM_FILE ="/lustre/storeA/users/josteinbl/H2O/wp4/TOPD/ISBA-TOPD-coupling/dtm50_6702_50m_33.tif"
    MIN_AREA = 2000

    dem = support.DEMFromTIF(DEM_FILE)
    dem = support.DEMFromTIF(DEM_FILE)
    geo = support.Geo("l2e", x_0, y_0, 500, 600, 50)
    support.TopdMapFromGeo(geo, dem).save_to_file("test_kise.map")
    support.TopdMapFromFile("test_kise.map").plot()
    # support.TopdMapFromFile("test.map").plot_map(plot_proj="utm33")
    # support.TopdMapFromFile("/lustre/storeA/users/josteinbl/H2O/wp4/TOPD/ISBA-TOPD-coupling/KISE_FilledDTM.map").plot()
    # support.TopdMapFromFile("/lustre/storeA/users/josteinbl/H2O/wp4/TOPD/ISBA-TOPD-coupling/KISE_FilledDTM.map").plot_map(data_is_l2e=False)

    '''
    CAT_FilledDTM.map
    CAT_connections.vec
    CAT_slope.vec
    CAT_RiverDist.map
    CAT_HillDist.map
    '''
