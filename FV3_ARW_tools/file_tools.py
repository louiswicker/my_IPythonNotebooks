import numpy as np
import netCDF4 as ncdf
import matplotlib as mlab
import matplotlib.pyplot as plt
import xarray as xr
import glob as glob
import os as os
import sys as sys

_Rgas       = 287.04
_gravity    = 9.806

# These are 45 vertical levels that the FV3 puts out - use them here to map ARW to that grid for comparison

plevels = np.asarray([100000.,  97500.,  95000.,  92500.,  90000.,  87500.,  85000.,  82500.,
                       80000.,  77500.,  75000.,  72500.,  70000.,  67500.,  65000.,  62500.,
                       60000.,  57500.,  55000.,  52500.,  50000.,  47500.,  45000.,  42500.,
                       40000.,  37500.,  35000.,  32500.,  30000.,  27500.,  25000.,  22500.,
                       20000.,  17500.,  15000.,  12500.,  10000.,   7000.,   5000.,   3000.,
                        2000.,   1000.,    700.,    500.,    200.])
nz_new = plevels.shape[0]

#--------------------------------------------------------------------------------------------------

def interp1d_np(data, z, zi):
#     print(f"data: {data.shape} | z: {z.shape} | zi: {zi.shape}")
    return np.interp(zi, z, data)

#--------------------------------------------------------------------------------------------------

def fv3_extract_variables_over_region(file, sw_corner=None, ne_corner=None, writeout=True, prefix=None):
    
    # get some stuff set up.

    temperature = ['TMP_P0_L100_GRLL0', 'TEMP']
    height      = ['HGT_P0_L100_GRLL0', 'HGT']
    sfc_height  = ["HGT_P0_L1_GRLL0", 'SFC_HGT']
    w           = ['DZDT_P0_L100_GRLL0','W']
    u           = ['UGRD_P0_L100_GRLL0', 'U']
    v           = ['VGRD_P0_L100_GRLL0', 'U']
    uh          = ['UPHL_P0_2L103_GRLL0','UH']
    ref_1km     = ['MAXREF_P8_L103_GRLL0_max1h', 'REF_1KM']
    cref        = ['REFC_P0_L200_GRLL0', 'CREF']

    print(f'-'*120,'\n')
    print(f'FV3_Extract: Extracting variables over region from input file: {file}','\n')
    
    ds_conus = xr.open_dataset(file)

    # Change some variable names for convienence

    ds_conus = ds_conus.rename_dims({'xgrid_0':'nx', 'ygrid_0': 'ny', 'lv_ISBL0': 'nz'})
    ds_conus = ds_conus.rename_vars({'lv_ISBL0':'levels', 'gridlat_0': 'lats', 'gridlon_0': 'lons'})    
    ds_conus = ds_conus.sortby('nz', ascending=False)
    
    # extract region
    
    if (sw_corner and len(sw_corner) > 1) and (ne_corner and len(ne_corner) > 1):
        lat_min = min(sw_corner[0], ne_corner[0])
        lat_max = max(sw_corner[0], ne_corner[0])
        lon_min = min(sw_corner[1], ne_corner[1])
        lon_max = max(sw_corner[1], ne_corner[1])
        print(f'Creating a sub-region of DataArray: {lat_min:.2f}, {lon_min:.2f}, {lat_max:.2f}, {lon_max:5.2f}','\n')

        ds_conus = ds_conus.where( (lat_min      < ds_conus.lats) & (ds_conus.lats < lat_max)
                                 & (lon_min+360. < ds_conus.lons) & (ds_conus.lons < lon_max+360.), drop=True)
        
    # I could just drop all the extra data sets out, but its probably faster for me to simply recreate a new separate one.
      
    ds_final = xr.Dataset(coords={"lats": (["ny","nx"], ds_conus['lats'].data),
                                  "lons": (["ny","nx"], ds_conus['lons'].data),  # important!
                                  "pres": (["nz"],      plevels)  } )
                       
    ds_final["lats"]    = (["ny", "nx"],        ds_conus['lats'].data )
    ds_final["lons"]    = (["ny", "nx"],        ds_conus['lons'].data )
    ds_final["SFC_HGT"] = (["ny", "nx"],        ds_conus[sfc_height[0]].data)
    ds_final["HGT"]     = (["nz", "ny", "nx"],  ds_conus[height[0]].data)
    ds_final["W"]       = (["nz", "ny", "nx"],  ds_conus[w[0]].values)
    ds_final["TEMP"]    = (["nz", "ny", "nx"],  ds_conus[temperature[0]].data)
    ds_final["U"]       = (["nz", "ny", "nx"],  ds_conus[u[0]].data)
    ds_final["V"]       = (["nz", "ny", "nx"],  ds_conus[v[0]].data)
    ds_final["CREF"]    = (["ny", "nx"],        ds_conus[cref[0]].data)
    ds_final["UH"]      = (["ny", "nx"],        ds_conus[uh[0]].data)
    ds_final["plevels"] = (["nz"], plevels)
             
    if writeout:
        dir, base = os.path.split(file)
        if ((sw_corner and len(sw_corner) > 1) and (ne_corner and len(ne_corner) > 1)):
            if prefix == None: 
                outfilename = os.path.join(dir, 'region%s' % base[4:])
                ds_final.to_netcdf(outfilename, mode='w')
                print(f'Successfully wrote new data to file:: {outfilename}','\n')
                return ds_final, outfilename
        else:
            if prefix == None:  
                outfilename = os.path.join(dir, 'full%s' % base[4:])
                ds_conus.to_netcdf(outfilename, mode='w')  
                print(f'Successfully wrote new data to file:: {outfilename}','\n')
                return ds_conus, outfilename

#--------------------------------------------------------------------------------------------------
    
def wrf_extract_variables_over_region(file, sw_corner=None, ne_corner=None, writeout=True, prefix=None):

    # get some stuff set up.

    omega       = ['VVEL_P0_L105_GLC0','OMEGA']
    pressure    = ['PRES_P0_L105_GLC0', 'PRES']
    temperature = ['TMP_P0_L105_GLC0', 'TEMP']
    height      = ["HGT_P0_L105_GLC0", 'HGT']
    sfc_height  = ["HGT_P0_L1_GLC0", 'SFC_HGT']
    uh          = ['MXUPHL_P8_2L103_GLC0_max1h','UH_MX']
    u           = ['UGRD_P0_L105_GLC0', 'U']
    v           = ['VGRD_P0_L105_GLC0', 'V']
    cref        = ['REFC_P0_L10_GLC0', 'CREF']

    print(f'-'*120,'\n')
    print(f'WRF_Extract: Extracting variables over region from input file: {file}','\n')
    
    ds_conus = xr.open_dataset(file)

    # Change some variable names for convienence

    ds_conus = ds_conus.rename_dims({'xgrid_0':'nx', 'ygrid_0': 'ny', 'lv_HYBL0': 'nz'})
    ds_conus = ds_conus.rename_vars({'lv_HYBL0':'levels', 'gridlat_0': 'lats', 'gridlon_0': 'lons'}) 

    # extract region drop the halo...

    if (sw_corner and len(sw_corner) > 1) and (ne_corner and len(ne_corner) > 1):
        lat_min = min(sw_corner[0], ne_corner[0])
        lat_max = max(sw_corner[0], ne_corner[0])
        lon_min = min(sw_corner[1], ne_corner[1])
        lon_max = max(sw_corner[1], ne_corner[1])
        
        print(f'Creating a sub-region of DataArray: {lat_min:.2f}, {lon_min:.2f}, {lat_max:.2f}, {lon_max:5.2f}','\n')

        ds_conus = ds_conus.where( (lat_min < ds_conus.lats) & (ds_conus.lats < lat_max)
                                 & (lon_min < ds_conus.lons) & (ds_conus.lons < lon_max), drop=True)
    
# Convert omega --> w

    w_hyb    = -ds_conus[omega[0]].data / ( (_gravity * ds_conus[pressure[0]].data) / (_Rgas * ds_conus[temperature[0]].data) )
    
# These variables will be interpolated to the plevels from 3D hybrid pressures in HRRR model

    p        = ds_conus[pressure[0]].values
    hgt_hyb  = ds_conus[height[0]].values
    temp_hyb = ds_conus[temperature[0]].values
    u_hyb    = ds_conus[u[0]].values
    v_hyb    = ds_conus[v[0]].values
            
# These variables will be interpolated to the plevels from 3D hybrid pressures

    w        = np.zeros((len(plevels),len(ds_conus.ny),len(ds_conus.nx)),dtype=np.float32)
    hgt      = np.zeros((len(plevels),len(ds_conus.ny),len(ds_conus.nx)),dtype=np.float32)
    temp     = np.zeros((len(plevels),len(ds_conus.ny),len(ds_conus.nx)),dtype=np.float32)
    u        = np.zeros((len(plevels),len(ds_conus.ny),len(ds_conus.nx)),dtype=np.float32)
    v        = np.zeros((len(plevels),len(ds_conus.ny),len(ds_conus.nx)),dtype=np.float32)

# This is gotta be the slow way to do this...

    for i in np.arange(len(ds_conus.nx)):
        for j in np.arange(len(ds_conus.ny)):

            w[::-1,j,i]    = interp1d_np(   w_hyb[::-1,j,i], p[::-1,j,i], plevels[::-1])
            hgt[::-1,j,i]  = interp1d_np( hgt_hyb[::-1,j,i], p[::-1,j,i], plevels[::-1])
            temp[::-1,j,i] = interp1d_np(temp_hyb[::-1,j,i], p[::-1,j,i], plevels[::-1])
            u[::-1,j,i]    = interp1d_np(u_hyb[::-1,j,i], p[::-1,j,i], plevels[::-1])
            v[::-1,j,i]    = interp1d_np(v_hyb[::-1,j,i], p[::-1,j,i], plevels[::-1])

    ds_final = xr.Dataset(coords={"lats": (["ny","nx"], ds_conus['lats'].data),
                                  "lons": (["ny","nx"], ds_conus['lons'].data),
                                  "pres": (["nz"],      plevels)  } )
                       
    ds_final["W"]       = (["nz", "ny", "nx"],    w[...])
    ds_final["HGT"]     = (["nz", "ny", "nx"],  hgt[...])
    ds_final["TEMP"]    = (["nz", "ny", "nx"], temp[...])
    ds_final["U"]       = (["nz", "ny", "nx"],    u[...])
    ds_final["V"]       = (["nz", "ny", "nx"],    v[...])
    ds_final["CREF"]    = (["ny", "nx"],       ds_conus[cref[0]].data)
    ds_final["UH"]      = (["ny", "nx"],       ds_conus[uh[0]].data[0])
    ds_final["SFC_HGT"] = (["ny", "nx"],       ds_conus[sfc_height[0]].data)
    ds_final["plevels"] = (["nz"], plevels)
    
    print(f'Successfully interpolated fields from file:  {file}','\n')
        
    if writeout:
        dir, base = os.path.split(file)
        if ((sw_corner and len(sw_corner) > 1) and (ne_corner and len(ne_corner) > 1)):
            if prefix == None: 
                outfilename = os.path.join(dir, 'region%s' % base[4:])
                ds_final.to_netcdf(outfilename, mode='w')
                print(f'Successfully wrote new data to file:: {outfilename}','\n')
                return ds_final, outfilename
        else:
            if prefix == None:  
                outfilename = os.path.join(dir, 'full%s' % base[4:])
                ds_conus.to_netcdf(outfilename, mode='w')  
                print(f'Successfully wrote new data to file:: {outfilename}','\n')
                return ds_conus, outfilename
            
    return None