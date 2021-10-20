#
import numpy as N
import matplotlib
import matplotlib.pyplot as P
from matplotlib.ticker import ScalarFormatter
from matplotlib import colors
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.ticker as ticker
from metpy.plots import ctables
#
from dataclasses import dataclass


_ref_norm, _ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)


@dataclass
class plot_spec:
    plot_type: str
    cmap:      Any = None
    clevels:   Any = None
    norm:      Any = None
    cint:      Any = None

def plot_spec_init():
    
    ref_norm, _ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)
    
    def __init__(self):

plot_defaults = {
                 'REF':{'cmap': _ref_cmap, 'clevels': None, norm': _ref_norm, 'plottype': 'pcolor'}
                }



#===============================================================================

def nearlyequal(a, b, sig_digit=7):
    """ Measures the equality (for two floats), in unit of decimal significant 
        figures.  If no sigificant digit is specified, default is 7 digits. """

    if a == b:
        return True

    difference = abs(a - b)
    avg = (a + b)/2
    
    return N.log10(avg / difference) >= sig_digit
    
#===============================================================================

def nice_mxmnintvl( dmin, dmax, outside=True, max_steps=15, cint=None, sym=False):
    """ Description: Given min and max values of a data domain and the maximum
                     number of steps desired, determines "nice" values of 
                     for endpoints and spacing to create a series of steps 
                     through the data domain. A flag controls whether the max 
                     and min are inside or outside the data range.
  
        In Args: float   dmin 		the minimum value of the domain
                 float   dmax       the maximum value of the domain
                 int     max_steps	the maximum number of steps desired
                 logical outside    controls whether return min/max fall just
                                    outside or just inside the data domaiN.
                     if outside: 
                         min_out <= min < min_out + step_size
                                         max_out >= max > max_out - step_size
                     if inside:
                         min_out >= min > min_out - step_size
                                         max_out <= max < max_out + step_size
      
                 float    cint      if specified, the contour interval is set 
                                    to this, and the max/min bounds, based on 
                                    "outside" are returned.

                 logical  sym       if True, set the max/min bounds to be anti-symmetric.
      
      
        Out Args: min_out     a "nice" minimum value
                  max_out     a "nice" maximum value  
                  step_size   a step value such that 
                                     (where n is an integer < max_steps):
                                      min_out + n * step_size == max_out 
                                      with no remainder 
      
        If max==min, or a contour interval cannot be computed, returns "None"
     
        Algorithm mimics the NCAR NCL lib "nice_mxmnintvl"; code adapted from 
        "nicevals.c" however, added the optional "cint" arg to facilitate user 
        specified specific interval.
     
        Lou Wicker, August 2009 """

    table = N.array([1.0,2.0,2.5,4.0,5.0,10.0,20.0,25.0,40.0,50.0,100.0,200.0,
                      250.0,400.0,500.0])

    if nearlyequal(dmax,dmin):
        return None
    
    # Help people like me who can never remember - flip max/min if inputted reversed
    if dmax < dmin:
        amax = dmin
        amin = dmax
    else:
        amax = dmax
        amin = dmin

    if sym:
        smax = max(amax.max(), amin.min())
        amax = smax
        amin = -smax

    d = 10.0**(N.floor(N.log10(amax - amin)) - 2.0)
    if cint == None or cint == 0.0:
        t = table * d
    else:
        t = cint
    if outside:
        am1 = N.floor(amin/t) * t
        ax1 = N.ceil(amax/t)  * t
        cints = (ax1 - am1) / t 
    else:
        am1 = N.ceil(amin/t) * t
        ax1 = N.floor(amax/t)  * t
        cints = (ax1 - am1) / t
    
    # DEBUG LINE BELOW
    #print(t, am1, ax1, cints)
    
    if cint == None or cint == 0.0:   
        try:
            index = N.where(cints < max_steps)[0][0]
            return am1[index], ax1[index], cints[index]
        except IndexError:
            return None
    else:
        return am1, ax1, cint

#===================================================================================================

def nice_clevels( *args, **kargs):
    """ Extra function to generate the array of contour levels for plotting 
        using "nice_mxmnintvl" code.  Removes an extra step.  Returns 4 args,
        with the 4th the array of contour levels.  The first three values
        are the same as "nice_mxmnintvl". """
    
    try:
        amin, amax, cint = nice_mxmnintvl(*args, **kargs)
        return amin, amax, cint, N.linspace(amin, amax, num=cint, endpoint=True) 
    except:
        return None

#===============================================================================

def plot_contour(fld, x = None, y = None, title = None, glat=None, glon=None,
                 clevels = None, cmap = None, zoom = None, scale = 1, ax = None, **kwargs):
   
    if x == None:
        x = scale*N.arange(fld.shape[1])

    if y == None:
        y = scale*N.arange(fld.shape[0])
        
    xx, yy = N.meshgrid(x, y)
        
    if title == None:
        title = '2D_FLD'
        
    plot_display = 'fcontour'
        
    if 'plot_defaults' in kwargs:
        for item, value in kwargs.get('plot_defaults'):
            
        
        if plot_spec in plot_defaults.keys():
            if plot_defaults[plot_spec][plottype] == 'pcolor':
                plot_display = 'pcolor'
                norm         =  plot_defaults[plot_spec]['norm']
            else:
                plot_display = 'fcontour'
                clevels = plot_defaults[plot_spec]['climits']
                cmap = plot_defaults[plot_spec]['cmap']
        else:
            print('The %s default plotting parameters are not set, using defaults')

    if type(clevels) == type(None):
        amin, amax, cint = nice_mxmnintvl(fld.min(), fld.max(), outside=True)
        clevels = N.linspace(amin, amax, num=int(cint), endpoint=True)
                      
    if type(cmap) == type(None):
        cmap = 'bwr'
        
    if ax == None:
        if plot_display == 'pcolor':
            plt.pcolor(xx, yy, fld, cmap=cmap, norm=norm)
            plot = P.contour(xx, yy,  fld, clevels[::2], colors='k', linewidths=0.5)
        else:
            plot = P.contourf(xx, yy, fld, clevels, cmap = cmap)
            plot = P.contour(xx, yy,  fld, clevels[::2], colors='k', linewidths=0.5)
    else:
        
        if plot_display == 'pcolor':
            plt.pcolor(xx, yy, fld, cmap=cmap, norm=norm)
        else:
            plot = ax.contourf(xx, yy, fld, clevels, cmap = cmap)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        P.colorbar(plot, cax=cax, orientation='vertical')

        plot = ax.contour(x, y,  fld, clevels[::2], colors='k', linewidths=0.5)

        ax.set_aspect('equal', 'datalim')
        ax.set_title(title, fontsize=10)

        if zoom:
            dx = xx.max() - xx.min()
            dy = yy.max() - yy.min()
            ax.set_xlim(xx.min() + dx*zoom[0],xx.min()+dx*zoom[1])
            ax.set_ylim(yy.min() + dy*zoom[2],yy.min()+dy*zoom[3])
            
        at = AnchoredText("Max: %4.1f \n Min: %4.1f" % (fld.max(),fld.min()),
                          loc=4, prop=dict(size=20), frameon=True,)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

        ax.add_artist(at)
        


    
# get coordinates for contour plots

#     cbar    = P.colorbar(plot,location='right',pad="5%")
#     cbar.set_label(title)
#     plot    = P.contour(x, y,  fld, clevel[::2], colors='k', linewidths=0.5)

#===============================================================================

def contour_intervals(fld, default=None):
    
    if default == None:
        amin, amax, cint = nice_mxmnintvl(fld.min(), fld.max(), outside=True, **kwargs)
        clevels = N.linspace(amin, amax, num=int(cint), endpoint=True)
        
        return
    
    elif default.upper() == 'DBZ' or default[0:3].upper() == 'REF':
        clevels = 10.* N.arange(8)
        