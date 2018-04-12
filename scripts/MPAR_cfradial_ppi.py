# Author: Jonathan J. Helmus (jhelmus@anl.gov)
# License: BSD 3 clause

import scipy
import copy
import matplotlib.pyplot as plt
import pyart
import matplotlib
import pylab as P
from scipy import signal
import numpy as N
import math
import sys
import netCDF4
from optparse import OptionParser
from netcdftime import utime, num2date
import os
import ctables
from mpl_toolkits.basemap import Basemap
from pyproj import Proj
import time as timeit
import numpy.ma as ma
import netCDF4 as ncdf
import glob

#---------------------------------------------------------------------------------------------------
# At bottom of script is a copy of the Fortran90 routine needed for the cressman analysis
# and the F2PY command to compile it

import cressman as cress


# Lambert conformal stuff
tlat1 = 35.3500
tlat2 = 35.8900
cen_lat = 35.6200
cen_lon = -97.9120

# Colorscale information
ref_scale = (0.,74.)
vr_scale  = (-40.,40.)

# Range rings in km
range_rings = [25, 50, 75, 100, 125]  

# Region to plot
sw_lat = 35.00
sw_lon = -98.75
ne_lat = 36.00
ne_lon = -97.5

_missing = -32768.

# Debug print stuff....
debug = False

#======================= EXAMPLE for Container CLASS ====================================
# class data_bin(object):
#     pass
#    new_obj = data_bin()
#    new_obj.fields = {}
#    new_obj.loc = copy.deepcopy(display_obj.loc)
#    new_obj.x = copy.deepcopy(display_obj.x)
#    new_obj.y = copy.deepcopy(display_obj.y)

#############################################################################################
def create_ppi_map(display_obj, plot_range_rings=True, ax=None, **kwargs):

   radar_lon = display_obj.loc[1]
   radar_lat = display_obj.loc[0]
   
   p1 = Proj(proj='lcc', ellps='WGS84', datum='WGS84', lat_1=tlat1, lat_2=tlat2, lon_0=radar_lon)

   x_offset, y_offset = p1(radar_lon, radar_lat)
   
   x = display_obj.x + x_offset
   y = display_obj.y + y_offset
   
   lon, lat = p1(x, y, inverse=True)
   
   map = mybasemap(lat, lon, sw_lat, sw_lon, ne_lat, ne_lon, shape_env=True, ax=ax, **kwargs)

   radar_x, radar_y = map(radar_lon, radar_lat)
   xmap, ymap = map(lon, lat)

   if plot_range_rings:
      angle = N.linspace(0., 2.0 * N.pi, 360)
      for ring in range_rings:    
         xpts = radar_x + ring * 1000. * N.sin(angle)
         ypts = radar_y + ring * 1000. * N.cos(angle)
         map.plot(xpts, ypts, color = 'gray', alpha = 0.5, linewidth = 1.0, ax=ax)   
          
   return map, xmap, ymap
#===============================================================================
def mybasemap(xlat, xlon, sw_lat, sw_lon, ne_lat, ne_lon, scale = 1.0, ticks = True, 
              resolution='c',area_thresh = 10., shape_env = False, 
              counties=False, states = False, lat_lines=True, lon_lines=True, 
              pickle = False, ax=None):

   tt = timeit.clock()

   map = Basemap(llcrnrlon=sw_lon, llcrnrlat=sw_lat, \
                  urcrnrlon=ne_lon, urcrnrlat=ne_lat, \
                  lat_1=0.5*(ne_lat+sw_lat), lon_0=0.5*(ne_lon+sw_lon), \
                  projection = 'lcc',      \
                  resolution=resolution,   \
                  area_thresh=area_thresh, \
                  suppress_ticks=ticks, ax=ax)

   if counties:
      map.drawcounties()
      
   if states:
       map.drawstates()
       
   if lat_lines == True:
       lat_lines = N.arange(sw_lat, ne_lat, 0.5)
       map.drawparallels(lat_lines, labels=[True, False, False, False])
       
   if lon_lines == True:
       lon_lines = N.arange(sw_lon, ne_lon, 0.5)
       map.drawmeridians(lon_lines, labels=[False, False, False, True])
       
   # Shape file stuff

   if shape_env:

      try:
         shapelist = os.getenv("PYESVIEWER_SHAPEFILES").split(":")

         if len(shapelist) > 0:
            for item in shapelist:
               items = item.split(",")
               shapefile  = items[0]
               color      = items[1]
               linewidth  = float(items[2])
               if debug:
                   print(linewidth)
                   print(items)
               s = map.readshapefile(shapefile,'counties',drawbounds=True,color=color,linewidth=linewidth)

            for shape in map.counties:
               xx, yy = zip(*shape)
               map.plot(xx,yy,color=color,linewidth=linewidth)

      except OSError:
         print "GIS_PLOT:  NO SHAPEFILE ENV VARIABLE FOUND "

   # pickle the class instance.

   if debug:  print(timeit.clock()-tt,' secs to create original Basemap instance')

   if pickle:
      pickle.dump(map,open('mymap.pickle','wb'),-1)
      print(timeit.clock()-tt,' secs to create original Basemap instance and pickle it')

   return map

#############################################################################################
def compute_azshear(display_obj, field='VEL', return_type=None):

   data = display_obj.fields[field]['data']
   
   az_shear = N.zeros(data.shape, dtype='float64')
      
   vr_shear_list = N.gradient(data)    # returns a list of gradient in N-dims
   
   vr_shear = ma.filled(vr_shear_list[0], 0.0)   # important step - get the az-derivative and convert....
   
   daz = N.radians(display_obj.azimuths[2:] - display_obj.azimuths[0:-2])
   
   dr = daz[...,None] * display_obj.ranges[None,...]
   
   az_shear[1:-1,:] = vr_shear[1:-1,:] / dr
   
   print("\nAZ_SHEAR calculation done:  Max shear:  %4.4f  Min shear:  %4.4f" % (az_shear.max(), az_shear.min()))
   
   tmp = ma.array(N.zeros(data.shape, dtype='float64'), mask=N.ones(data.shape, dtype=bool), fill_value=_missing) 
   tmp.data[...] = N.reshape(az_shear, (data.shape[0], data.shape[1]))
   tmp.mask[...] = N.reshape(N.where(az_shear != 0.0, False, True), (data.shape[0], data.shape[1]))
   display_obj.fields['AZ_SHEAR'] = {'data': tmp}
   
   if return_type == None:
       return display_obj
   else:																				# return data, coordinates, and radar location - this is enough info to map
       return {'data': tmp, 'x': display_obj.x, 'y': display_obj.y, 'z': display_obj.z,
               'radar_lat': display_obj.loc[0], 'radar_lon': display_obj.loc[1], }

#############################################################################################
def plot_ppi_map(display_obj, field, cmap=ctables.Carbone42, vRange=None, var_label = None, 
                 plot_range_rings=True, ax=None):

   map, x, y = create_ppi_map(display_obj, plot_range_rings=True, ax=ax)
   
   if vRange == None:
       vRange = display_obj._parse_vmin_vmax(field, vmin, vmax)
          
   data = display_obj.fields[field]['data'][...]
   
   plot = map.pcolormesh(x, y, data, cmap=cmap, vmin = vRange[0], vmax = vRange[1], ax=ax)
   cbar = map.colorbar(plot, location = 'right', pad = '3%')
   
   if var_label:
       cbar.set_label(var_label, fontweight='bold')
   else:
       cbar.set_label(field, fontweight='bold')

   title_string = display_obj.generate_title(field, (display_obj.elevations[0].round(2))).replace('T','--')
   
   if var_label:
       title_string = title_string.replace(field, var_label)
       
   P.title(title_string)
   
   print("\nCompleted plot for %s" % field)
     
   return display_obj.generate_filename('',display_obj.elevations[0].round(2)).replace("__","_")
     
#############################################################################################
def regrid_plot_map(display_obj, xgrid, ygrid, field = 'REF', var_label="DATA", cmap=ctables.Carbone42,
                    vRange=[-0.01,0.1], roi = 4000., plot_range_rings=True, ax=None, file=False):

# Set up map and projection

   radar_lon = display_obj.loc[1]
   radar_lat = display_obj.loc[0]
   
   map, xplot, yplot = create_ppi_map(display_obj, plot_range_rings=True, ax=ax)
   
   radar_x, radar_y = map(radar_lon, radar_lat)

# Cressman analyze the observed 2D field to a new grid

   obs = display_obj.fields[field]['data'][...]
   
   mask = obs.mask[...]
   notmask = N.where(mask == False)
   
   obsM  = obs.data[notmask]
   xobsM = xplot[notmask]
   yobsM = yplot[notmask]
   
   xgrid = radar_x + 1000.*N.arange(xgrid[0], xgrid[1]+xgrid[2], xgrid[2])

   ygrid = radar_y + 1000.*N.arange(ygrid[0], ygrid[1]+ygrid[2], ygrid[2])

   new = cress.cressman(xobsM, yobsM, obsM, xgrid, ygrid, roi)

# Now plot the analyzed fields
   
   if vRange == None:
       vRange = (new.min(), new.max())
       
   plot = map.pcolormesh(xgrid, ygrid, new.transpose(), cmap=cmap, vmin = vRange[0], vmax = vRange[1], ax=ax)
   cbar = map.colorbar(plot, location = 'right', pad = '3%')
   cbar.set_label(var_label, fontweight='bold')

   title_string = display_obj.generate_title(field, (display_obj.elevations[0].round(2))).replace('T','-')
   title_string = title_string.replace("Deg.", "deg")
   if var_label:
       title_string = title_string.replace(field, var_label)
       
   P.title(title_string)
   
   print("\nCompleted plot for %s" % field)

   print xgrid.shape, ygrid.shape
   if file:

# create the fileput filename and create new netCDF4 file
     filename = title_string.replace(" ","_") + ".nc"
     filename = filename.replace("\n", "")
     filename = filename.replace("deg_", "")
     filename = filename.replace("-", "",2)     
     filename = filename.replace("__", "_")
     root = ncdf.Dataset(filename, 'w', format='NETCDF4')
     root.createDimension('X', xgrid.shape[0])
     root.createDimension('Y', ygrid.shape[0])
     var2d = root.createVariable(field.replace(" ","_"), 'f4', ('Y','Y'), fill_value='-9999.')
     x1d = root.createVariable('X', 'f4', ('X',), fill_value='-9999.')
     y1d = root.createVariable('Y', 'f4', ('Y',), fill_value='-9999.')
     x1d[:] = xgrid[:]
     y1d[:] = ygrid[:]
     var2d[...] = new[...]
     root.close()

#############################################################################################
# MAIN program

files = glob.glob("./MPAR_cfradial_data/20110524/cfrad.*_el0.51_PPI.nc")
print files

output_dir = ""

for filename in files:

		display_obj = pyart.graph.RadarDisplay(pyart.io.read_cfradial(filename))
				
		fig, axes = P.subplots(2, 2, sharey=True, figsize=(12,12))

		outfile  = plot_ppi_map(display_obj, 'DZ', vRange=ref_scale, cmap=ctables.NWSRef, ax=axes[0,0], var_label='Reflectivity')

		outfile  = plot_ppi_map(display_obj, 'DV', vRange=(-40,40), cmap=ctables.Carbone42, ax=axes[0,1], var_label='Radial Velocity')

		display_obj = compute_azshear(display_obj, field='DV')

		outfile = plot_ppi_map(display_obj, 'AZ_SHEAR', vRange=(-0.02,0.02), ax=axes[1,0], var_label='Radial Velocity')
			
		az_shear_plot = regrid_plot_map(display_obj, (-360,0,2), (-180,180,2), field='AZ_SHEAR', roi=1666., 
																																		var_label="2KM Analyzed AZ Shear", vRange=(-0.006,0.006), ax=axes[1,1], file=True)

		fig.subplots_adjust(left=0.06, right=0.90, top=0.93, bottom=0.03, wspace=0.35)

		P.savefig(os.path.join(output_dir, outfile), format="png", dpi=300)

#P.show()

