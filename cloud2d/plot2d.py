import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredText


#-----------------------------------------------------------------------------
# Plot function for 2D plots

def plotXZ(x, z, fld, clevels, **kwargs):

    # See if you are passing in an axis
    if "ax" in kwargs:
        ax = kwargs['ax']
    else:
        plt.figure(figsize = (8,8))
        ax = plt.subplot(111)

    # See if you need to substract a basestate        
    if "basestate" in kwargs:
        basestate = kwargs['basestate']
        if basestate.ndim < 2:
            nz = fld.shape[0]
            pltfld = fld - basestate.reshape(nz,1)
        else:
            pltfld = fld - basestate
    else:
        pltfld = fld.copy()

    # Check for a colormap being passed in  
    if "cmap" in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = cm.jet
          
    plotf   = ax.contourf(x, z, pltfld, clevels, cmap=cmap)
    plot    = ax.contour(x, z, pltfld, clevels[::2], color='k')
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(plotf, cax=cax)

    at = AnchoredText("Fld_Max: %4.1f \nFld_Min: %4.1f" % (pltfld.max(), pltfld.min()), 
                      loc=2, prop=dict(size=10), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    # Process title args    
    if "main" in kwargs:
        ax.set_title(kwargs['main'])

    if "ylabel" in kwargs:
        ax.set_ylabel(kwargs['ylabel'])

    if "zlabel" in kwargs:
        ax.set_ylabel(kwargs['zlabel'])

    if "xlabel" in kwargs:
        ax.set_xlabel(kwargs['xlabel']) 

    # See if we should save the plot to a file
    if "saveplot" in kwargs:
        plt.savefig(kwargs['saveplot'])    
      
    # See if we should show the plot  
    if "showplot" in kwargs:
        plt.show()      
        
