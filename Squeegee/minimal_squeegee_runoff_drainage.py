import sys, os, glob
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, shape, Point
import geopandas as gp
import xesmf as xe
import pyproj
import requests
import cartopy.crs as ccrs
from cartopy.io.img_tiles import Stamen
tiler = Stamen('terrain-background')

def KGEf(s,e):
    cc = np.corrcoef(e,s)[0,1]
    sdr = np.nanstd(s)/np.nanstd(e)
    mr = np.nanmean(s)/np.nanmean(e)
    KGE = np.round(1. - np.sqrt((cc-1)**2 + (sdr -1)**2 + (mr-1)**2),5)
    return KGE, cc, sdr, mr


def convert_3D_2D(geometry):
    '''
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
    '''
    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == 'Polygon':
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == 'MultiPolygon':
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(MultiPolygon(new_multi_p))
    return new_geo


expdirs='/lustre/storeB/project/nwp/H2O/wp4/SURFEX_offline/open_SURFEX_V8_1/MY_RUN/KTEST/'

# folder where runs are, this could have been a long list of different exps.
expdirfs=['527_450/']
# start of nc-file-output from runs
ncstart=  ['openofl']
# choose a name for legend in plots
expnam=['mebglac_albevol']

numruns=len(expnam)

#assuming same forcing
FORCING=xr.open_dataset('/lustre/storeB/project/nwp/H2O/wp4/FORCING/527_450_2020/FORCING_527_450_202010.nc', cache=False,use_cftime=False)
print(FORCING.LAT.min(), FORCING.LAT.max())
print(FORCING.LON.min(), FORCING.LON.max())

fPREP=xr.open_dataset(expdirs+'/527_450/openprepn.nc', cache=False,use_cftime=False)
fPGD=xr.open_dataset(expdirs+'/527_450/openpgd.nc', cache=False,use_cftime=False)


#add coords etc so that surfex files may be processed
R_pysurfex=6371000 # and met nordic
R_arome   =6371229
Rpy=6.371229e+06
proj_string = "+proj=lcc +lat_0=63 +lon_0=15 +lat_1=63 +lat_2=63 +units=m +no_defs +R=" + str(Rpy)
myP=pyproj.Proj(proj_string)

lcc = ccrs.LambertConformal(#globe=globe, # no datumshift in GCMs
                            central_longitude=15, central_latitude=63,
                            standard_parallels=(63,63))#,
geodetic=ccrs.Geodetic() #default WGS84




#read a list of stations
stationst='/lustre/storeB/project/nwp/H2O/wp4/RRdata/stations_Huang20.txt'
stations=pd.read_csv(stationst, header=0,index_col=0,sep=None,
                     dtype={'GauID': 'str'})
stations=stations.drop_duplicates(subset=('GauID'), keep='last')

# Nr.	GauID	Lat	Long	Ara_km2	Group
daily = 1440
station='2.25.0'# stations.GauID[2]

stations['GauID']=stations.GauID+'.0'


#read shape files with polygons
#Outles to sea defined by NVE provided in latlon coords
#regs2 = gp.read_file('/lustre/storeB/project/nwp/H2O/wp4/RRdata/shapefiles/latlonREG/Nedborfelt/Nedborfelt_Vassdragsomr.shp')

# or Outlets to sea defined for Europe CCM2
ccm2_basins = gp.read_file('/lustre/storeB/project/nwp/H2O/wp4/RRdata/shapefiles/CCM2/ccm21/WGS84_W2008.gdb',layer='SEAOUTLETS')
#ccm2sub = ccm2_basins.cx[FORCING.LON.min().values:FORCING.LON.max().values,FORCING.LAT.min().values:FORCING.LAT.max().values]    

# or cathements where NVE have measuring stations
#regs = gp.read_file('/lustre/storeB/project/nwp/H2O/wp4/RRdata/shapefiles/utm33shp/NVEData/Hydrologi/Hydrologi_TotalNedborfeltMalestasjon.shp')
regs2 = gp.read_file('/lustre/storeB/project/nwp/H2O/wp4/RRdata/shapefiles/latlonHyd/latlonhydorder/NVEData/Hydrologi/Hydrologi_TotalNedborfeltMalestasjon.shp')

#inspect the shape file
print(regs2.crs)
print(regs2.columns)

#remove extra dim in nve shape-files
regs2.geometry = convert_3D_2D(regs2.geometry)

plot_polygons=False
#make a plot of the polygons:
if plot_polygons==True:
    
    fig = plt.figure(figsize=(4,8))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(), frameon=False)
 
    ax1.coastlines(resolution='10m')
  
    regs2.plot(column='stID',kind='geo',edgecolor="g",linewidth=0.2,
               ax=ax1,legend=True,cmap='Reds',transform=ccrs.Geodetic())
    #,legend_kwds={"orientation": "horizontal", "pad": 0.01, "label": "VassOmrNr"})
    #ccm2sub.plot(column='AREA_KM2',kind='geo',edgecolor="g",linewidth=0.2,ax=ax1,legend=True,cmap='Reds',legend_kwds={"orientation": "horizontal", "pad": 0.01})
    ax1.axis('off')
    plt.title("")
    plt.savefig("polygons.png")
    plt.show()




#-----------------------------------------
# read data in this way because a large amount of data
variables = ['RUNOFFC_ISBA', 'DRAINC_ISBA']
def preprocess(ds):
    return ds[variables]

dictds={}

for i in np.arange(0,numruns):
    dictds[expnam[i]] = xr.open_mfdataset(expdirs+expdirfs[i]+ncstart[i]+
                                          'sel2021??.nc',
                                          cache=False,
                                          preprocess=preprocess,
                                          concat_dim="time",
                                          combine="nested",
                                          chunks={'time': 1})
    print(expnam[i] + ' given to: ' +expdirs+expdirfs[i]+ncstart[i]+
          'sel20????.nc')


# xesmf needs dataset to generate weights
TSr=dictds[expnam[-1]]['RUNOFFC_ISBA'].isel(time=0).to_dataset()

xfalseaR, yfalseaR = myP(fPREP.LONORI.data, fPREP.LATORI.data,inverse=False)
#Surfex makes all coords positive by setting the origin to the lower left corner

TSr=TSr.assign_coords(x= fPREP.XX[0,:].data+xfalseaR)
TSr=TSr.assign_coords(y= fPREP.YY[:,0].data+yfalseaR)
TSr = TSr.rename({'xx': 'x', 'yy': 'y'})

dx, dy = fPGD.DX.data[0,0], fPGD.DY.data[0,0]
Xcorners=np.arange(TSr['x'].data[0]-dx/2., TSr['x'].data[-1]+3*dx/2., dx)
Ycorners=np.arange(TSr['y'].data[0]-dy/2., TSr['y'].data[-1]+3*dy/2., dy)
#Lon2, Lat2 = myP(fPREP.XX.data+xfalseaR,fPREP.YY.data+yfalseaR,inverse=True)
Lon2b, Lat2b = myP(*np.meshgrid(Xcorners,Ycorners),inverse=True) 

TSr.coords['xb'] = (Xcorners)
TSr.coords['yb'] = (Ycorners)
TSr.coords['lat_b'] = (('yb','xb'),Lat2b)
TSr.coords['lon_b'] = (('yb','xb'),Lon2b)
TSr.set_coords(['lat_b','lon_b'])

TSr.coords['lat'] = (('y','x'),FORCING.LAT.data)
TSr.coords['lon'] = (('y','x'),FORCING.LON.data)
TSr.lon.attrs=FORCING.LON.attrs
TSr.lat.attrs=FORCING.LAT.attrs
TSr.set_coords(['lat','lon'])

TSr['mask']=TSr.RUNOFFC_ISBA.isnull()
TSr['mask'].values=np.where(~TSr.RUNOFFC_ISBA.isnull(),1,0)

TSr.attrs['pyproj_srs']=proj_string


#make an outer domain polygon to crop the input shape file

dompoly=Polygon(zip([TSr['lon_b'][0,0].data,TSr['lon_b'][0,-1].data,TSr['lon_b'][-1,-1].data,
                     TSr['lon_b'][-1,0].data],[TSr['lat_b'][0,0].data,TSr['lat_b'][0,-1].data,
                                               TSr['lat_b'][-1,-1].data,TSr['lat_b'][-1,0].data]))

gdom = gp.GeoSeries([dompoly])

# assing defined polygon to a new dataframe
pol_gpd= gp.GeoDataFrame()
pol_gpd['geometry'] = None
pol_gpd.loc[0,'geometry'] = dompoly
pol_gpd.crs=regs2.crs #hope okay

# crop shape file to domain, remove basins > 2 km2 
result = gp.sjoin(regs2, pol_gpd, how='inner')#, op='within')
resultn=result[result.areal_km2 > 2]
resultn=resultn[resultn.stID!='2.11.0'] #only missing data 
resultn.to_csv('basins_used.csv')

# This generates the weights, takes time the first time only
# ...
savg = xe.SpatialAverager(TSr, resultn.geometry, geom_dim_name="stID",
                          filename='/lustre/storeB/users/josteinbl/TOPD/spatial_avg_4catchments_527nature2lim.nc',reuse_weights=True)

plotweights=False

# this takes a lot of memory for all these basins, not recommended
if plotweights:
    w = xr.DataArray(
    savg.weights.toarray().reshape(resultn.geometry.size, *TSr.lat.shape),
    dims=("stID", *TSr.lat.dims),
    coords=dict(stID=out.stID, **TSr.lon.coords),
    )

    plt.subplots_adjust(top=0.9)
    facets = w.plot(col="stID", col_wrap=6, aspect=2, vmin=0, vmax=0.05)
    facets.cbar.set_label("Averaging weights")
    plt.savefig('weights.png')

#store the dfs in dict

# here the sparse matrix with grid cell weights for each ploygon is generated
# see xesmf doc: 
# https://pangeo-xesmf.readthedocs.io/en/latest/notebooks/Spatial_Averaging.html
# the matrix can be stored as a nc-file and re-used, so that the program runs 
# very fast once it is generated. 
# The weights are specific to the domain, projection, resolution, and to 
# options like 
#&nam_pgd_arrange_cover 
#  lwater_to_nature = .true. 
#  ltown_to_rock = .true.
#  which alters where RUNOFFC etc is defined 

cachrunoff={}

for i in np.arange(0,numruns):

    tmp1=dictds[expnam[i]]['RUNOFFC_ISBA'] +dictds[expnam[i]]['DRAINC_ISBA']
    
    #some experiments have accumlated runoff (i.e. do not use option
    # LRESETCUMUL = .true. in NAM_WRITE_DIAG_SURFn 
    # see https://www.umr-cnrm.fr/surfex/spip.php?article406 )
    # so need to deaccumulate
    if expnam[i] in ['nm12snowl', 'nmdtB07','nmSOC','nbni','s']:
        tmp1=tmp1.diff(dim='time')
    tmp1=tmp1.to_dataset(name=expnam[i])
    tmp1['mask']=TSr.RUNOFFC_ISBA.isnull()
    tmp1['mask'].values=np.where(~TSr.RUNOFFC_ISBA.isnull(),1,0)
    out=savg(tmp1[expnam[i]])
    #out = out.assign_coords(roms_id=xr.DataArray(resultn.index.values, dims=("roms_id",)))
    out = out.assign_coords(stID=xr.DataArray(resultn["stID"], dims=("stID",)))
    print("out")
    print(out)
    cachrunoff[expnam[i]]=out.to_pandas()
    rainfRun= cachrunoff[expnam[i]]=out.to_pandas()
    print("rainfRun")
    print(rainfRun)
    rainfRun["50.64.0"].plot()
    plt.show()
    cachrunoff[expnam[i]].to_csv(expnam[i]+'.csv')

