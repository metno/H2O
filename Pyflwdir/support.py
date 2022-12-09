"""H2O support library."""
import logging
import rasterio
from rasterio.mask import mask
from rasterio import features
from pyproj import Proj, Transformer
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from cartopy.crs import LambertConformal, PlateCarree, UTM, epsg, Globe
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
from gridpp import Grid, nearest
import numpy as np
import pyflwdir


class Projection():
    """Projection."""

    def __init__(self, name):
        """Constuct projection object.

        Args:
            name (str): Projection name

        Raises:
            NotImplementedError: _description_
        """
        logging.info("Construct projection name=%s", name)
        self.proj_name = name
        proj_string = None

        if name == "wgs84":
            proj_string = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
            crs = PlateCarree()
        elif name == "wgs84_epsg":
            proj_string = "EPSG:4326"
            crs = PlateCarree()
        elif name == "l2e":
            # proj_string = "+proj=lcc +lat_1=46.8 +lat_0=46.8 +lon_0=0 +k_0=0.99987742 " \
            # "+x_0=600000 +y_0=2200000 +ellps=clrk80ign +pm=paris +units=m +no_defs +type=crs"
            proj_string = "+proj=lcc +lat_1=46.8 +lat_0=46.8 +lon_0=0 +k_0=0.99987742 +x_0=600000 "\
                          "+y_0=2200000 +ellps=clrk80ign +pm=paris +towgs84=-168,-60,320,0,0,0,0 "\
                           "+units=m +no_defs +type=crs"
            prime_meridian = 2.33722917
            scale_factor = 0.99987742
            scale_factor = None
            globe = Globe(
                ellipse="clrk80ign",
                flattening=scale_factor
            )
            # Geodetic CRS: NTF (Paris)
            # Datum: Nouvelle Triangulation Francaise (Paris)
            # Ellipsoid: Clarke 1880 (IGN)
            crs = LambertConformal(
                central_latitude=46.8,
                standard_parallels=[46.8, 46.8],
                central_longitude=prime_meridian,
                false_easting=600000,
                false_northing=2200000,
                globe=globe
            )
        elif name == "l2e_epsg":
            proj_string = "EPSG:27572"
            # crs = LambertConformal(central_latitude=63.3, central_longitude=15.0)
            crs = epsg(27572)
        elif name == "metcoop":
            proj_string = "+proj=lcc +lat_0=63.3 +lon_0=15 +lat_1=63.3 " \
                          "+lat_2=63.3 +no_defs +R=6.371e+06"
            # cartopy.crs.LambertConformal
            crs = LambertConformal(central_latitude=63.3, central_longitude=15.0)
        elif name == "utm33_epsg":
            proj_string =  "EPSG:32633"
            crs = UTM(zone="33")
        elif name == "utm33":
            proj_string = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs +type=crs"
            crs = UTM(zone="33")
        else:
            raise NotImplementedError(name)
        self.proj = Proj(proj_string)
        self.cartopy_crs = crs

    def transform(self, proj_out, xvals, yvals):
        """Tranform from one CRS to another.

        Args:
            proj_out (_type_): _description_
            xvals (_type_): _description_
            yvals (_type_): _description_

        Returns:
            _type_: _description_
        """
        transformer = Transformer.from_crs(self.proj.crs, proj_out.proj.crs, always_xy=True)
        xvals, yvals = transformer.transform(xvals, yvals)  # noqa
        return xvals, yvals


class LonLatGrid(Grid):
    """Encapsulate grid."""

    def __init__(self, lons, lats):
        logging.info("lons.shape %s", lons.shape)
        logging.info("lats.shape %s", lats.shape)
        Grid.__init__(self, *(lons, lats))
        self.lons = lons
        self.lats = lats
        self.n_x = lons.shape[1]
        self.n_y = lats.shape[0]
        self.npoints = self.n_x * self.n_y

    def plot_lon_lat_grid(self, proj):

        ax1 = plt.axes(projection=proj.cartopy_crs)
        ax1.coastlines("10m")

        plt.scatter(self.lons, self.lats, transform=PlateCarree())
        plt.text(-4.87, 41.31, "ll", transform=PlateCarree())
        plt.text(-4.87, 51.14, "ul", transform=PlateCarree())
        plt.text(9.63, 41.31, "lr", transform=PlateCarree())
        plt.text(9.63, 51.14, "ur", transform=PlateCarree())
        plt.text(8.775383, 58.461541, "ArendalP", transform=PlateCarree())
        plt.text(9.243745, 58.723588, "RisorP", transform=PlateCarree())
        plt.text(9.415991, 58.869724, "KrageroP", transform=PlateCarree())
        plt.show()


class Geo(Projection, LonLatGrid):
    """Geometry."""

    def __init__(self, proj, x_0, y_0, n_x, n_y, msize):
        Projection.__init__(self, proj)
        logging.info("x_0=%s y_0=%s", x_0, y_0)
        self.x_0 = x_0
        self.y_0 = y_0
        self.n_x = n_x
        self.n_y = n_y
        self.msize = msize
        xvals = []
        yvals = []
        for i in range(0, self.n_x):
            xvals.append(self.x_0 + (float(i) * self.msize))
        for i in range(0, self.n_y):
            yvals.append(self.y_0 + (float(i) * self.msize))

        self.xvals, self.yvals= np.meshgrid(np.array(xvals), np.array(yvals))
        wgs84 = Projection("wgs84")
        lons, lats = self.transform(wgs84, self.xvals, self.yvals)
        print(lons.shape)
        print(lats.shape)
        LonLatGrid.__init__(self, lons, lats)

    def plot_map(self, field=None):
        """Plot map.

        Args:
            field (_type_, optional): _description_. Defaults to None.
        """

        ax1 = plt.axes(projection=self.cartopy_crs)
        ax1.coastlines("10m")

        print(self.xvals, self.yvals)
        if field is None:
            plt.scatter(self.xvals, self.yvals)
        else:
            plt.contourf(self.xvals, self.yvals, field)
        plt.title(str(self.proj_name))
        plt.text(-4.87, 41.31, "ll", transform=PlateCarree())
        plt.text(-4.87, 51.14, "ul", transform=PlateCarree())
        plt.text(9.63, 41.31, "lr", transform=PlateCarree())
        plt.text(9.63, 51.14, "ur", transform=PlateCarree())
        plt.text(8.775383, 58.461541, "ArendalP", transform=PlateCarree())
        plt.text(9.243745, 58.723588, "RisorP", transform=PlateCarree())
        plt.text(9.415991, 58.869724, "KrageroP", transform=PlateCarree())
        plt.show()


class DEM(Projection, LonLatGrid):
    """Get DEM from TIF."""

    def __init__(self, src):
        self.src = src
        elevtn = src.read()
        self.elevtn = elevtn[0, :, :]
        self.n_x = self.elevtn.shape[0]
        self.n_y = self.elevtn.shape[1]
        nodata = src.nodata
        self.src_transform = src.transform
        self.src_crs = src.crs
        extent = np.array(src.bounds)[[0, 2, 1, 3]]
        self.x_0 = float(extent[0])
        self.y_0 = float(extent[2])
        self.x_1 = float(extent[1])
        self.y_1 = float(extent[3])
        self.d_x = ((extent[1] - extent[0])/self.n_x)
        self.d_y = ((extent[3] - extent[2])/self.n_y)
        Projection.__init__(self, "utm33")
        self.latlon = self.proj.crs.is_geographic
        prof = src.profile
        self.nodata = nodata
        self.extent = extent
        self.prof = prof
        self.mask = None
        xvals = []
        yvals = []
        for i in range(0, self.n_x):
            xvals.append(self.x_0 + (float(i) * self.d_x))
        for i in range(0, self.n_y):
            yvals.append(self.y_0 + (float(i) * self.d_y))
        self.xvals, self.yvals = np.meshgrid(np.array(xvals), np.array(yvals))
        wgs84 = Projection("wgs84")
        lons, lats = self.transform(wgs84, self.xvals, self.yvals)
        LonLatGrid.__init__(self, lons, lats)

    def plot_map(self):
        """Plot map."""

        ax1 = plt.axes(projection=self.cartopy_crs)
        ax1.coastlines("10m")

        values = self.elevtn[::-1,:]
        values = np.ma.masked_equal(values, self.nodata)
        plt.contourf(self.xvals, self.yvals, values)
        plt.text(-4.87, 41.31, "ll", transform=PlateCarree())
        plt.text(-4.87, 51.14, "ul", transform=PlateCarree())
        plt.text(9.63, 41.31, "lr", transform=PlateCarree())
        plt.text(9.63, 51.14, "ur", transform=PlateCarree())
        plt.text(8.775383, 58.461541, "ArendalP", transform=PlateCarree())
        plt.text(9.243745, 58.723588, "RisorP", transform=PlateCarree())
        plt.text(9.415991, 58.869724, "KrageroP", transform=PlateCarree())
        plt.title(str(self.proj_name))
        plt.show()


class MaskedDEM(DEM):
    """Get Maskes DEM from rasterio src."""

    def __init__(self, src, regs2sub):
        DEM.__init__(self, src)
        regs2sub = regs2sub.to_crs(crs=self.proj.crs)
        out_image, __ = mask(self.src, regs2sub.geometry, filled = True)

        out_image=out_image[0,:,:]
        out_image[out_image==0] = self.nodata
        self.mask = (out_image == self.nodata)
        self.elevtn = np.ma.masked_equal(out_image, self.nodata)
        # itemindex = np.where(not np.isnan(self.elevtn))
        # print(itemindex)
        print(self.elevtn)
        print(self.mask)


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


class TopdMap(Projection, LonLatGrid):
    """TOPD map class."""

    def __init__(self, geo, nodata, minval, maxval, values):
        """Construct object.

        Args:
            geo (Geo): Geometry
            nodata (float): No data value.
            minval (float): Min value
            maxval (float): Max value
            values (float): Values
        """
        self.geo = geo
        self.nodata = nodata
        self.minval = minval
        self.maxval = maxval
        self.values = values
        self.mask = None
        Projection.__init__(self, "l2e")

        wgs84 = Projection("wgs84")
        lons, lats = self.transform(wgs84, geo.xvals, geo.yvals)
        LonLatGrid.__init__(self, lons, lats)


    def plot_map(self, title=None):
        """PLot TOPD map.

        Args:
            title (str, optional): Title. Defaults to None.
        """

        ax1 = plt.axes(projection=self.cartopy_crs)
        ax1.coastlines("10m")

        values = self.values
        values = np.ma.masked_equal(values, self.nodata)
        plt.contourf(self.geo.xvals, self.geo.yvals, values, cmap="gist_earth_r",
            alpha=0.5,
            origin='upper',
            vmin=0,
            vmax=1000)
        plt.text(-4.87, 41.31, "ll", transform=PlateCarree())
        plt.text(-4.87, 51.14, "ul", transform=PlateCarree())
        plt.text(9.63, 41.31, "lr", transform=PlateCarree())
        plt.text(9.63, 51.14, "ur", transform=PlateCarree())
        plt.text(8.775383, 58.461541, "ArendalP", transform=PlateCarree())
        plt.text(9.243745, 58.723588, "RisorP", transform=PlateCarree())
        plt.text(9.415991, 58.869724, "KrageroP", transform=PlateCarree())
        if title is not None:
            plt.title(title)
        fig = plt.gcf()
        cax = fig.add_axes([0.8, 0.27, 0.02, 0.12])
        plt.colorbar(cax=cax, orientation="vertical", extend="max")
        cax.set_ylabel("elevation [m] NHM50 L2E")
        plt.show()

    def save_to_file(self, fname):
        """Save TOPD to file.

        Args:
            fname (str): File name
        """
        logging.info("Save file %s", fname)
        nrows0 = 6
        nrows1 = 8
        header_vals = [
            self.geo.x_0,
            self.geo.y_0,
            self.n_x,
            self.n_y,
            self.nodata,
            self.geo.msize,
            self.minval,
            self.maxval
        ]
        with open(fname, mode="w", encoding="utf-8") as file_handler:
            for __ in range(0, nrows0):
                file_handler.write("\n")
            for i in range(0, nrows1):
                print(i, header_vals[i])
                file_handler.write(f"{header_vals[i]}\n")
            self.values[np.isnan(self.values)] = self.nodata
            # np.savetxt(file_handler, np.transpose(self.values).flatten(), fmt='%1.3f')
            np.savetxt(file_handler, self.values.flatten(), fmt='%1.3f')


class TopdMapFromFile(TopdMap):
    """TOPD map file from file."""

    def __init__(self, fname):
        """Construct object.

        Args:
            fname (str): File name.
        """
        geo, nodata, minval, maxval, values = self.read_file(fname)
        TopdMap.__init__(self, geo, nodata, minval, maxval, values)

    def read_file(self, fname):
        """Read file.

        Args:
            fname (str): File name.

        Returns:
            tuple: geo, nodata, minval, maxval, values
        """
        logging.info("Read file %s", fname)
        nrows0 = 6
        nrows1 = 8
        header_vals = []
        with open(fname, mode="r", encoding="utf-8") as file_handler:
            for __ in range(0, nrows0):
                line = file_handler.readline()
            header_vals = []
            for __ in range(0, nrows1):
                line = file_handler.readline()
                val = line.split()[0]
                header_vals.append(val)

        x_0 = float(header_vals[0])
        y_0 = float(header_vals[1])
        n_x = int(header_vals[2])
        n_y = int(header_vals[3])
        nodata = float(header_vals[4])
        msize = float(header_vals[5])
        geo = Geo("l2e", x_0, y_0, n_x, n_y, msize)
        minval = float(header_vals[6])
        maxval = float(header_vals[7])
        values = np.loadtxt(fname, skiprows=(nrows0 + nrows1))
        values = values.reshape(n_y, n_x)
        values = np.ma.masked_equal(values, nodata)
        return geo, nodata, minval, maxval, values


class TopdMapFromGeo(TopdMap):
    """TOPD map from a geo object."""

    def __init__(self, geo, dem, values=None):

        if values is None:
            values = dem.elevtn[::-1,:]

        values[values == dem.nodata] = np.nan
        values = np.ma.masked_equal(values, dem.nodata)
        values = nearest(dem, geo, values)
        logging.info("values before: %s", values)
        for i in range(0, geo.n_x):
            for j in range(0, geo.n_y):
                lon = geo.lons[j,i]
                lat = geo.lats[j,i]
                num_neighbours = dem.get_num_neighbours(lon, lat, geo.msize * 1.1)
                if num_neighbours == 0:
                    values[j, i] = np.nan
        values = np.ma.masked_equal(values, np.nan)
        values = np.ma.masked_equal(values, dem.nodata)
        logging.info("values after: %s", values)

        minval = values.min()
        maxval = values.max()
        nodata = dem.nodata
        TopdMap.__init__(self, geo, nodata, minval, maxval, values)


class SubBasin():
    """Calculate sub-basin."""

    def __init__(self, flw_obj, dem, min_area=2000):
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
        gdf_subbas = vectorize(
            dem, subbas.astype(np.int32), 0, flw_obj.transform, dem.crs, name="basin"
        )
        # randomize index for visualization
        basids = gdf_subbas["basin"].values
        gdf_subbas["color"] = np.random.choice(basids, size=basids.size, replace=False)
        self.gdf_subbas = gdf_subbas


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
    for index, __ in regs2.iterrows():
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
    utm33 = Projection("utm33")
    ax1 = fig.add_subplot(1, 1, 1, projection=utm33.cartopy_crs, frameon=False)
    ax1.coastlines(resolution='10m')
    hdatametasub.plot(
        column='Name',kind='geo',edgecolor="g",linewidth=0.2,ax=ax1,
        legend=True,cmap='tab20',
        transform=utm33.cartopy_crs
    )
    #,legend_kwds={"orientation": "horizontal", "pad": 0.01})#, "label": "Name"})
    regs2sub.geometry.boundary.plot(
        edgecolor="b",color=None,linewidth=0.2,
        ax=ax1,transform=utm33.cartopy_crs
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
        transform=dem.src_transform,
        latlon=dem.latlon
    )


def pyflwdir_from_array(flwdir, crs, transform, cache=True,):
    """Create a plwdir object from DEM.

    Args:
        flwdir (_type_): _description_
        crs (_type_): _description_
        transform (_type_): _description_
        cache (bool, optional): _description_. Defaults to True.

    Returns:
        FlwDir: PyFlwdir object

    """
    return pyflwdir.from_array(
        flwdir,
        ftype="d8",
        transform=transform,
        latlon=crs.is_geographic,
        cache=cache,
    )


def pyflwdir_from_d8_file(d8_file, cache=True):
    """Create a plwdir object from d8 file.

    Args:
        d8_file (str): Geotiff D8 file

    Returns:
        FlwDir: PyFlwdir object

    """
    with rasterio.open(d8_file, "r") as src:
        flwdir = src.read(1)
        crs = src.crs
        transform = src.transform
        return pyflwdir_from_array(flwdir, crs, transform, cache=cache)


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
    gdf = gpd.GeoDataFrame.from_features(feats, crs=dem.src_crs)
    # create nice colormap of Blues with less white
    cmap_streams = colors.ListedColormap(cm.Blues(np.linspace(0.4, 1, 7)))
    gdf_plot_kwds = dict(column="strord", cmap=cmap_streams)
    # plot streams with hillshade from elevation data (see utils.py)

    fig = plt.figure(figsize=(8, 12))
    ax1 = fig.add_subplot(projection=dem.cartopy_crs)
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


def plot_sub_basin(gdf_subbas, dem, title=None, file=None):
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
    ax1 = fig.add_subplot(projection=dem.cartopy_crs)

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
    gdf_subbas = vectorize(dem, subbas.astype(np.int32), 0,
                           flw_obj.transform, dem.src_crs, name="basin")
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


def vectorize(dem, data, nodata, transform, crs, name="value"):
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
    print(data)
    if dem.mask is not None:
        data = np.ma.masked_where(dem.mask, data)
    print("Masked", data)
    feats_gen = features.shapes(
        data,
        # mask=dem.mask,
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
        dem.elevtn, nodata=dem.nodata, latlon=dem.latlon, transform=dem.src_transform
    )
    slope = vectorize(dem, slope.astype(np.int32), 0, dem.src_transform, dem.src_crs, name="slope")
    return slope


def plot_slope(slope, hillshading, dem, title=None, file=None):
    """Plot streams.

    Args:
        slopes (_type_): _description_
        hillshading (_type_): _description_
        dem (DEM): _description_
        title (_type_, optional): _description_. Defaults to None.
        file (_type_, optional): _description_. Defaults to None.

    """
    fig = plt.figure(figsize=(8, 12))
    ax1 = fig.add_subplot(projection=dem.cartopy_crs)
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
    projection = Projection("wgs84")
    ax1 = fig.add_subplot(projection=projection.cartopy_crs)
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


def stream_distance(flw_obj, dem, unit='cell'):
    """_summary_

    Args:
        flw_obj (_type_): _description_
        dem (_type_): _description_
        unit (str, optional): _description_. Defaults to 'cell'.

    Returns:
        _type_: _description_
    """
    mask_array = dem.mask
    distance = flw_obj.stream_distance(mask=mask_array, unit=unit)
    print(distance)
    distance = vectorize(
        dem, distance.astype(np.int32), -9999, dem.src_transform,
        dem.src_crs, name="distance"
    )
    print(distance)
    return distance


def plot_stream_distance(distance, dem, hillshading, title=None, file=None):
    """Plot streams.

    Args:
        slopes (_type_): _description_
        dem (SEM): _description_
        title (_type_, optional): _description_. Defaults to None.
        file (_type_, optional): _description_. Defaults to None.

    """
    fig = plt.figure(figsize=(8, 12))
    ax1 = fig.add_subplot(projection=dem.cartopy_crs)
    ax1.imshow(
                hillshading,
                origin="upper",
                extent=dem.extent,
                cmap="Greys",
                alpha=0.3,
                zorder=0,
    )
    print(distance)
    distance.plot(ax=ax1, cmap='hot', legend=True)

    if title is not None:
        plt.title(title)
    if file is not None:
        plt.savefig(file)
    plt.show()

'''
def topd_write_river_distance(dem, distance, filename=None):

    if filename is not None:
        np.savetxt(filename, np.transpose([distance]), fmt='%1.3f')


def topd_write_connections(filename=None):
    if filename is not None:
        np.savetxt(filename, np.transpose([zxidx, zval, waterrec, waterrec, upslopeidx, waterrec,
                                           default, default, default, default, default, default,
                                           default, default, default, default, default, default,
                                           default, default]), fmt='%1.3f')


def topd_write_slope(filename=None):

    if filename is not None:
        np.savetxt(filename, np.transpose(
                                         [zxidx, tanbList, default, default, default, default, twi]
                                         ),
                   fmt='%i %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f')
'''
