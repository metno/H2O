"""H2O support library."""
import rasterio
from rasterio.mask import mask
from rasterio import features
import numpy as np
import pyflwdir
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon


class DEM():
    """Get DEM from TIF."""

    def __init__(self, src):
        self.src = src
        elevtn = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        extent = np.array(src.bounds)[[0, 2, 1, 3]]
        latlon = src.crs.is_geographic
        prof = src.profile
        self.elevtn = elevtn
        self.nodata = nodata
        self.transform = transform
        self.crs = crs
        self.extent = extent
        self.latlon = latlon
        self.prof = prof

    def plot(self, show=True, title=None, filename=None):
        """Plot DEM

        Args:
            show (bool, optional): _description_. Defaults to True.
            title (_type_, optional): _description_. Defaults to None.
            filename (_type_, optional): _description_. Defaults to None.

        """
        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(projection=ccrs.UTM(zone=33))
        im1 = ax1.imshow(
            self.elevtn,
            extent=self.extent,
            cmap="gist_earth_r",
            alpha=0.5,
            origin='upper',
            vmin=0,
            vmax=1000,
        )
        if title is not None:
            plt.title(title)
        fig = plt.gcf()
        cax = fig.add_axes([0.8, 0.27, 0.02, 0.12])
        fig.colorbar(im1, cax=cax, orientation="vertical", extend="max")
        cax.set_ylabel("elevation [m] NHM50 UTM33")
        if filename is not None:
            plt.savefig(filename, dpi=225)
        if show:
            plt.show()


class MaskedDEM(DEM):
    """Get Maskes DEM from rasterio src."""

    def __init__(self, src, regs2sub):
        DEM.__init__(self, src)
        regs2sub = regs2sub.to_crs(crs=self.crs)
        out_image, __ = mask(self.src, regs2sub.geometry, filled = True)
        out_image=out_image[0,:,:]
        out_image[out_image==0] = self.nodata
        self.elevtn = np.ma.masked_equal(out_image, self.nodata)


class DEMFromTIF(DEM):
    """Get DEM from TIF."""

    def __init__(self, tif_file):
        with rasterio.open(tif_file, "r") as src:
            DEM.__init__(self, src)


class MaskedDEMFromTIF(MaskedDEM):
    """Get Masked DEM from TIF file."""

    def __init__(self, tif_file, regs2sub):
        with rasterio.open(tif_file, "r") as src:
            MaskedDEM.__init__(self, src, regs2sub)


def get_dtm_areas(dtm_areas):
    """Get info about DTMs

    Args:
        dtm_areas (str): Filename

    Returns:
        geopandas: Read information
    """

    hdatameta = gpd.read_file(dtm_areas)
    return hdatameta


def regs2subcatchments(all_catchments):
    """Get a geopanda object of catchments.

    Args:
        all_catchments (str): File with catchments

    """
    def convert_3d_2d(geometry):
        """ Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons.

        Args:
            geometry (geopandas.Geometry): Geometry

        Returns:
            list: List with points
        """
        new_geo = []
        for point in geometry:
            if point.has_z:
                if point.geom_type == 'Polygon':
                    lines = [xy[:2] for xy in list(point.exterior.coords)]
                    new_p = Polygon(lines)
                    new_geo.append(new_p)
                elif point.geom_type == 'MultiPolygon':
                    new_multi_p = []
                    for a_p in point.geoms:
                        lines = [xy[:2] for xy in list(a_p.exterior.coords)]
                        new_p = Polygon(lines)
                        new_multi_p.append(new_p)
                    new_geo.append(MultiPolygon(new_multi_p))
        return new_geo

    regs2 = gpd.read_file(all_catchments)
    for index, poi in regs2.iterrows():
        print(regs2.loc[index, "vassOmrNr"], regs2.loc[index, "vassOmr"])
    regs2sub_catchment = regs2.iloc[:]
    geometry = convert_3d_2d(regs2sub_catchment.geometry)
    regs2sub_catchment.set_geometry(geometry)
    return regs2sub_catchment


def select_catchments(regs2sub, selection):
    """Select the catchments.

    Args:
        regs2sub (geopandas): Geopandas with catchments
        selection (list): List with strings defining catchments

    Returns:
        gepandas: Subset geopanda with selected catchments

    """
    dissolve = []
    for __, row in regs2sub.iterrows():
        val = "no"
        for catchment in selection:
            if row["vassOmrNr"] == catchment:
                val = "yes"
        dissolve.append(val)
    regs2sub_dissolve = regs2sub
    regs2sub_dissolve['dissolve'] = dissolve
    regs2sub_selection = regs2sub_dissolve.loc[regs2sub_dissolve["dissolve"] == "yes"]
    print(regs2sub_selection)
    return regs2sub_selection


def plot_dtm_info(regs2sub, hdatametasub, title=None):
    """Plot DTMs and catchments

    Args:
        regs2sub (_type_): _description_
        hdatametasub (_type_): _description_
        title (_type_, optional): _description_. Defaults to None.
    """
    fig = plt.figure(figsize=(4,8))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.UTM(zone=33), frameon=False)
    ax1.coastlines(resolution='10m')
    hdatametasub.plot(
        column='Name',kind='geo',edgecolor="g",linewidth=0.2,ax=ax1,
        legend=True,cmap='tab20',
        transform=ccrs.UTM(zone=33)
    )
    #,legend_kwds={"orientation": "horizontal", "pad": 0.01})#, "label": "Name"})
    regs2sub.geometry.boundary.plot(
        edgecolor="b",color=None,linewidth=0.2,
        ax=ax1,transform=ccrs.UTM(zone=33)
    )
    ax1.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()


def pyflwdir_from_dem(dem):
    """Create a plwdir object from DEM.

    Args:
        dem (DTM): Digital elevation model

    Returns:
        FlwDirRaster: PyFlwdir object

    """
    # returns FlwDirRaster object
    return pyflwdir.from_dem(
        data=dem.elevtn,
        nodata=dem.nodata,
        transform=dem.transform,
        latlon=dem.latlon
    )


def plot_streams(flwdir_object, hillshading, dem, title=None, file=None):
    """Plot streams.

    Args:
        flwdir_object (_type_): _description_
        hillshading (_type_): _description_
        dem (DEM): _description_
        title (_type_, optional): _description_. Defaults to None.
        file (_type_, optional): _description_. Defaults to None.

    """
    feats = flwdir_object.streams(min_sto=4)
    gdf = gpd.GeoDataFrame.from_features(feats, crs=dem.crs)
    # create nice colormap of Blues with less white
    cmap_streams = colors.ListedColormap(cm.Blues(np.linspace(0.4, 1, 7)))
    gdf_plot_kwds = dict(column="strord", cmap=cmap_streams)
    # plot streams with hillshade from elevation data (see utils.py)

    fig = plt.figure(figsize=(8, 12))
    ax1 = fig.add_subplot(projection=ccrs.UTM(zone=33))
    ax1.imshow(
                hillshading,
                origin="upper",
                extent=dem.extent,
                cmap="Greys",
                alpha=0.3,
                zorder=0,
            )
    # plot geopandas GeoDataFrame
    for gdf, kwargs in [(gdf, gdf_plot_kwds)]:
        gdf.plot(ax=ax1, **kwargs)

    if title is not None:
        plt.title(title)
    if file is not None:
        plt.savefig(file)
    plt.show()


def plot_sub_basin(gdf_subbas, title=None, file=None):
    """Plot sub basin.

    Args:
        gdf_subbas (_type_): _description_
        title (_type_, optional): _description_. Defaults to None.
        file (_type_, optional): _description_. Defaults to None.

    """
    # plot
    gpd_plot_kwds = dict(
        column="color", cmap=cm.Set3, edgecolor="black", alpha=0.6, linewidth=0.5, aspect=1)
    fig = plt.figure(figsize=(8, 12))
    ax1 = fig.add_subplot(projection=ccrs.UTM(zone=33))

    # plot geopandas GeoDataFrame
    for gdf, kwargs in [(gdf_subbas, gpd_plot_kwds)]:
        gdf.plot(ax=ax1, **kwargs)

    if title is not None:
        plt.title(title)
    if file is None:
        plt.savefig(file)
    plt.show()


def calc_sub_basin(flw_obj, dem, min_area=2000):
    """calculate subbasins with a minimum stream order 7 and its outlets.

    Args:
        flw_obj (_type_): _description_
        dem (DEM): DEM
        min_area (int, optional): _description_. Defaults to 2000.

    Returns:
        geopandas: _description_

    """
    subbas, __ = flw_obj.subbasins_area(min_area)
    # subbas, idxs_out = flw_obj.subbasins_area(min_area)

    # transfrom map and point locations to GeoDataFrames
    gdf_subbas = vectorize(subbas.astype(np.int32), 0, flw_obj.transform, dem.crs, name="basin")
    # randomize index for visualization
    basids = gdf_subbas["basin"].values
    gdf_subbas["color"] = np.random.choice(basids, size=basids.size, replace=False)
    return gdf_subbas


def write_d8_tif_file(flw_obj, dem, file):
    """Write tif file.

    Args:
        flw_obj (_type_): _description_
        dem (DEM): _description_
        file (str): File name.
    """
    d8_data = flw_obj.to_array(ftype="d8")
    # update data type and nodata value properties which are different compared to
    # the input elevation grid and write to geotif
    dem.prof.update(dtype=d8_data.dtype, nodata=247)
    with rasterio.open(file, "w", **dem.prof) as src:
        src.write(d8_data, 1)


def vectorize(data, nodata, transform, crs, name="value"):
    """Vectorize data

    Args:
        data (_type_): _description_
        nodata (_type_): _description_
        transform (_type_): _description_
        crs (_type_): _description_
        name (str, optional): _description_. Defaults to "value".

    Returns:
        _type_: _description_
    """
    feats_gen = features.shapes(
        data,
        mask=data != nodata,
        transform=transform,
        connectivity=8,
    )
    feats = [
        {"geometry": geom, "properties": {name: val}} for geom, val in list(feats_gen)
    ]

    # parse to geopandas for plotting / writing to file
    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    gdf[name] = gdf[name].astype(data.dtype)
    return gdf


def get_slope(dem):
    """Get slope.

    Args:
        flw_obj (_type_): _description_
        dem (DEM): _description_

    Returns:
        _type_: _description_

    """
    slope = pyflwdir.dem.slope(
        dem.elevtn, nodata=dem.nodata, latlon=dem.latlon, transform=dem.transform
    )
    slope = vectorize(slope.astype(np.int32), 0, dem.transform, dem.crs, name="slope")
    return slope


def plot_slope(slope, hillshading, dem, title=None, file=None):
    """Plot streams.

    Args:
        slopes (_type_): _description_
        hillshading (_type_): _description_
        dem (SEM): _description_
        title (_type_, optional): _description_. Defaults to None.
        file (_type_, optional): _description_. Defaults to None.

    """
    fig = plt.figure(figsize=(8, 12))
    ax1 = fig.add_subplot(projection=ccrs.UTM(zone=33))
    ax1.imshow(
                hillshading,
                origin="upper",
                extent=dem.extent,
                cmap="Greys",
                alpha=0.3,
                zorder=0,
    )
    slope.plot(ax=ax1)

    if title is not None:
        plt.title(title)
    if file is not None:
        plt.savefig(file)
    plt.show()


def quickplot(gdfs, raster=None, hillshade=True, extent=None, h_s=None, title=None, filename=None):
    """convenience method for plotting.

    Args:
        gdfs (_type_): _description_
        raster (_type_, optional): _description_. Defaults to None.
        hillshade (bool, optional): _description_. Defaults to True.
        extent (_type_, optional): _description_. Defaults to None.
        h_s (_type_, optional): _description_. Defaults to None.
        title (_type_, optional): _description_. Defaults to None.
        filename (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    fig = plt.figure(figsize=(8, 15))
    ax1 = fig.add_subplot(projection=ccrs.PlateCarree())
    # plot hillshade background
    if hillshade:
        ax1.imshow(
            h_s,
            origin="upper",
            extent=extent,
            cmap="Greys",
            alpha=0.3,
            zorder=0,
        )
    # plot geopandas GeoDataFrame
    for gdf, kwargs in gdfs:
        gdf.plot(ax=ax1, **kwargs)
    if raster is not None:
        data, nodata, kwargs = raster
        ax1.imshow(
            np.ma.masked_equal(data, nodata),
            origin="upper",
            extent=extent,
            **kwargs,
        )
    ax1.set_aspect("equal")
    if title is not None:
        ax1.set_title(title, fontsize="large")
    ax1.text(
        0.01, 0.01, "created with pyflwdir", transform=ax1.transAxes, fontsize="large"
    )
    if filename is not None:
        plt.savefig(f"{filename}.png")
    return ax1

