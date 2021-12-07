import numpy as np
import netCDF4 as ncdf
import matplotlib as mlab
import matplotlib.pyplot as plt
import xarray as xr
import glob as glob
import os as os
import sys as sys
import pygrib
import scipy.signal


_nthreads = 2

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
# Interp from 3D pressure to 1D pressure (convert from hybrid to constant p-levels)

def interp3d_np(data, p3d, p1d, nthreads = _nthreads):
    
    dinterp = np.zeros((len(p1d),data.shape[1],data.shape[2]),dtype=np.float32)

    if nthreads < 0:  # turning this off for now.
        def worker(i):
            print("running %d %s" % (i, data.shape))
            for j in np.arange(data.shape[1]):
                  dinterp[:,j,i] = np.interp(p1d[::-1], p3d[:,j,i], data[:,j,i])

        pool = mp.Pool(nthreads)
        for i in np.arange(data.shape[2]):
            pool.apply_async(worker, args = (i, ))
        pool.close()
        pool.join()
        
        return dinterp[::-1,:,:]
    
    else:        
        for i in np.arange(data.shape[2]):
            for j in np.arange(data.shape[1]):
                dinterp[:,j,i] = np.interp(p1d[::-1], p3d[:,j,i], data[:,j,i])
        
        return dinterp[::-1,:,:]
    
#--------------------------------------------------------------------------------------------------
# Plotting strings for filtered field labels
    
def title_string(time, pres, label, wmax, wmin, eps=None):
    if eps:
        return ("%2.2i UTC %s at Pres=%3.0f mb with EPS=%5.1f \n Wmax: %3.1f        Wmin: %4.2f" % (time, label, pres/100., eps, wmax, wmin))
    else:
        return ("%2.2i UTC %s at Pres=%3.0f mb \n Wmax: %3.1f        Wmin: %4.2f" % (time, label, pres/100., wmax, wmin))

#--------------------------------------------------------------------------------------------------
# Choose a section of the grid based on lat/lon corners - excludes the rest of grid from xarray

def extract_subregion(xr_obj, sw_corner=None, ne_corner=None, drop=True):
    
    if (sw_corner and len(sw_corner) > 1) and (ne_corner and len(ne_corner) > 1):
        lat_min = min(sw_corner[0], ne_corner[0])
        lat_max = max(sw_corner[0], ne_corner[0])
        lon_min = min(sw_corner[1], ne_corner[1])
        lon_max = max(sw_corner[1], ne_corner[1])
        
        print(f'Creating a sub-region of grid: {lat_min:.2f}, {lon_min:.2f}, {lat_max:.2f}, {lon_max:5.2f}','\n')

        return xr_obj.where( (lat_min < xr_obj.lats) & (xr_obj.lats < lat_max)
                           & (lon_min < xr_obj.lons) & (xr_obj.lons < lon_max), drop=drop)
    else:
        print(f"No grid information supplied - returning original grid!\n")
        return xr_obj

#--------------------------------------------------------------------------------------------------
# Special thanks to Scott Ellis of DOE for sharing codes for reading grib2

def grbFile_attr(grb_file):

    dataloc = np.array(grb_file[1].latlons())

    return np.float32(dataloc[0]), np.float32(dataloc[1])

def grbVar_to_slice(grb_obj, type=None):

    """Takes a single grb object for a variable returns a 2D plane"""

    return {'data' : np.float32(grb_obj[0].values), 'units' : grb_obj[0]['units'],
            'date' : grb_obj[0].date, 'fcstStart' : grb_obj[0].time, 'fcstTime' : grb_obj[0].step}

def grbVar_to_cube(grb_obj, type='isobaricInhPa'):

    """Takes a single grb object for a variable containing multiple
    levels. Can sort on type. Compiles to a cube"""

    all_levels = np.array([grb_element['level'] for grb_element in grb_obj])
    types      = np.array([grb_element['typeOfLevel'] for grb_element in grb_obj])

    if type != None:
        levels = []
        for n, its_type in enumerate(types):
            if type == types[n]:
                levels.append(all_levels[n])
        levels = np.asarray(levels)
    else:
        levels = all_levels

    n_levels   = len(levels)
    indexes    = np.argsort(levels)[::-1] # highest pressure first
    cube       = np.zeros([n_levels, grb_obj[0].values.shape[0], grb_obj[1].values.shape[1]])

    for k in range(n_levels):
        cube[k,:,:] = grb_obj[indexes[k]].values

    return {'data' : np.float32(cube), 'units' : grb_obj[0]['units'], 'levels' : levels[indexes],
            'date' : grb_obj[0].date, 'fcstStart' : grb_obj[0].time, 'fcstTime' : grb_obj[0].step}

#--------------------------------------------------------------------------------------------------

def hrrr_grib_read_variable(file, sw_corner=None, ne_corner=None, var_list=[''], interpP=True, writeout=True, prefix=None):
    
    # Special thanks to Scott Ellis of DOE for sharing codes for reading grib2
    
    default = {            #  Grib2 name                 / No. of Dims /  Type
               'TEMP':     ['Temperature',                           3, 'hybrid'],
               'OMEGA':    ['Vertical velocity',                     3, 'hybrid'],
               'U':        ['U component of wind',                   3, 'hybrid'],
               'V':        ['V component of wind',                   3, 'hybrid'],
               'UH':       [1018,                                    2, 'hybrid'],
               'CREF':     [1008,                                    2, 'hybrid'],
               }

    if var_list != ['']:
        variables = {k: default[k] for k in default.keys() & set(var_list)}  # yea, I stole this....
    else:
        variables = default

    if prefix == None:
        prefix = 'hrrr'

    print(f'-'*120,'\n')
    print(f'HRRR_Extract: Extracting variables from grib file: {file}','\n')

    # open file

    grb_file = pygrib.open(file)

    # Get lat lons

    lats, lons = grbFile_attr(grb_file)

    pres = None
    

    if interpP:  # need to extract out 3D pressure for interp.
        
        grb_var = grb_file.select(name='Pressure')
        cube = grbVar_to_cube(grb_var, type='hybrid')
        p3d  = cube['data']
        print(f'InterpP is True, Read 3D pressure field from GRIB file\n')
        print(f'P-max:  %5.2f  P-min:  %5.2f\n' % (p3d.max(), p3d.min()))

    for n, key in enumerate(variables):

        print('Reading my variable: ',key, 'from GRIB file variable: ',variables[key][0])

        if type(variables[key][0]) == type('1'):
            grb_var = grb_file.select(name=variables[key][0])
        else:
            grb_var = [grb_file.message(variables[key][0])]

        if variables[key][1] == 3:

            cube = grbVar_to_cube(grb_var, type=variables[key][2])
            
            if interpP:
                
                cubeI = interp3d_np(cube['data'], p3d, plevels)
                    
                new = xr.DataArray( cubeI, dims = ['nz','ny','nx'], 
                                                  coords={"lats": (["ny","nx"], lats),
                                                          "lons": (["ny","nx"], lons), 
                                                          "pres": (["nz"],      plevels) } )
                
            else:

                new = xr.DataArray( cube['data'], dims = ['nz','ny','nx'], 
                                                  coords={"lats": (["ny","nx"], lats),
                                                          "lons": (["ny","nx"], lons), 
                                                          "hybid": (["nz"],     cube['levels']) } )
            date      = cube['date'] 
            fcstStart = cube['fcstStart']
            fcstHour  = cube['fcstTime']

        if variables[key][1] == 2:

            cube = grbVar_to_slice(grb_var, type=variables[key][2])

            new = xr.DataArray( cube['data'], dims=['ny','nx'], 
                                             coords={"lats": (["ny","nx"], lats),
                                                     "lons": (["ny","nx"], lons) } )  
            date      = cube['date'] 
            fcstStart = cube['fcstStart']
            fcstHour  = cube['fcstTime']

        if n == 0:

            ds_conus = new.to_dataset(name = key)

        else:         

            ds_conus[key] = new

        del(new)

    # clean up grib file
    
    grb_file.close()
            
    # Convert omega --> w
    
    pp3 = np.broadcast_to(plevels, (p3d.shape[2],p3d.shape[1],len(plevels))).transpose()
    
    w_new = -ds_conus['OMEGA'].data / ( (_gravity * pp3 ) / (_Rgas * ds_conus['TEMP'].data) )
    
    ds_conus['W'] = xr.DataArray( w_new, dims = ['nz','ny','nx'], 
                                  coords={"lats": (["ny","nx"], lats),
                                          "lons": (["ny","nx"], lons), 
                                          "pres": (["nz"],      plevels) } )   
    # extract region
    
    ds_conus = extract_subregion(ds_conus, sw_corner=sw_corner, ne_corner=ne_corner)


    if writeout:
        dir, base = os.path.split(file)
        outfilename = os.path.join(dir, '%s_%8.8i%2.2i_F%2.2i.nc' % (prefix, date,fcstStart,fcstHour))
        ds_conus.to_netcdf(outfilename, mode='w')  
        print(f'Successfully wrote new data to file:: {outfilename}','\n')
        return ds_conus, outfilename     
    else:
        return ds_conus, outfilename
    
#--------------------------------------------------------------------------------------------------

def fv3_grib_read_variable(file, sw_corner=None, ne_corner=None, var_list=[''], writeout=True, prefix=None):
    
    # Special thanks to Scott Ellis of DOE for sharing codes for reading grib2
    
    default = {             #  Grib2 name                 / No. of Dims /  Type
               'TEMP':     ['Temperature',                 3, 'isobaricInPa'],
               'SFC_HGT':  ['Orography',                   2, 'isobaricInPa'],
               'HGT':      ['Geopotential Height',         3, 'isobaricInPa'],              
               'W':        ['Geometric vertical velocity', 3, 'isobaricInPa'],
               'U':        ['U component of wind',         3, 'isobaricInPa'],
               'V':        ['V component of wind',         3, 'isobaricInPa'],
               'UH':       ['Updraft Helicity',            2, 'isobaricInPa'],
               'CREF':     ['Derived radar reflectivity',  3, 'hybrid'      ],
               }

    if var_list != ['']:
        variables = {k: default[k] for k in default.keys() & set(var_list)}  # yea, I stole this....
    else:
        variables = default

    if prefix == None:
        prefix = 'rrfs'

    print(f'-'*120,'\n')
    print(f'FV3_Extract: Extracting variables from grib file: {file}','\n')

    # open file

    grb_file = pygrib.open(file)

    # Get lat lons

    lats, lons = grbFile_attr(grb_file)

    pres = [None]

    for n, key in enumerate(variables):

        print('Reading my variable: ',key, 'from GRIB variable: \n',variables[key][0])

        if type(variables[key][0]) == type('1'):
            grb_var = grb_file.select(name=variables[key][0])
        else:
            grb_var = [grb_file.message(variables[key][0])]
        
        if variables[key][1] == 3:

            if key == 'CREF':  # this is a painful hack because of grib weirdness
                cube = grbVar_to_cube(grb_var, type=None)
                new = xr.DataArray( cube['data'].max(axis=0), dims=['ny','nx'], 
                                    coords={"lats": (["ny","nx"], lats),
                                            "lons": (["ny","nx"], lons) } )
            else:
                cube = grbVar_to_cube(grb_var, type='isobaricInhPa')
                pres = cube['levels']
               
                new = xr.DataArray( cube['data'], dims=['nz','ny','nx'], coords={'pres': (['nz'], pres),
                                                                                 "lons": (["ny","nx"], lons),
                                                                                 "lats": (["ny","nx"], lats)} )      
        if variables[key][1] == 2:
            cube = grbVar_to_slice(grb_var)
            new = xr.DataArray( cube['data'], dims=['ny','nx'], coords={"lons": (["ny","nx"], lons),
                                                                        "lats": (["ny","nx"], lats)} )      

        if n == 0:
            
            ds_conus = new.to_dataset(name = key)
        else:
            ds_conus[key] = new
            
        del(new)
        
        date      = cube['date'] 
        fcstStart = cube['fcstStart']
        fcstHour  = cube['fcstTime']

    # clean up grib file
    
    grb_file.close()
    
    # extract region
    
    if (sw_corner and len(sw_corner) > 1) and (ne_corner and len(ne_corner) > 1):
        lat_min = min(sw_corner[0], ne_corner[0])
        lat_max = max(sw_corner[0], ne_corner[0])
        lon_min = min(sw_corner[1], ne_corner[1])
        lon_max = max(sw_corner[1], ne_corner[1])
        print(f'Creating a sub-region of conus grid: {lat_min:.2f}, {lon_min:.2f}, {lat_max:.2f}, {lon_max:5.2f}','\n')

        ds_conus = ds_conus.where( (lat_min < ds_conus.lats) & (ds_conus.lats < lat_max)
                                 & (lon_min < ds_conus.lons) & (ds_conus.lons < lon_max), drop=True)            
    if writeout:
        
        dir, base = os.path.split(file)
        outfilename = os.path.join(dir, '%s_%8.8i%2.2i_F%2.2i.nc' % (prefix, date,fcstStart,fcstHour))
        ds_conus.to_netcdf(outfilename, mode='w')  
        print(f'Successfully wrote new data to file:: {outfilename}','\n')
        
    else:
        return ds_conus, outfilename



#--------------------------------------------------------------------------------------------------

def fv3_ncdf_read_variable(file, sw_corner=None, ne_corner=None, writeout=True, prefix=None):
    
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
    
def hrrr_ncdf_read_variable(file, sw_corner=None, ne_corner=None, writeout=True, prefix=None):

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




