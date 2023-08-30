from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
from astropy.coordinates import SkyCoord
from astropy.table import Table
from numpy.polynomial import polynomial
from matplotlib.patches import Ellipse

import astropy.units as u
import numpy as np
import pylab as py
import pdb
import matplotlib.pyplot as plt
import healpy as hp
import scipy.ndimage as nd
import movies as mv


def synthpop(dist, feh=-2.0, age=1e10, AKs=0.0, mass=1e5):
    '''
    Parameters:
    dist: in kpc
    age: in years
    AKs: extinction in mags
    '''
    
    #Define parameters of stellar population
    logAge = np.log10(age) # Age in log(years)
    AKs = 0.0 # extinction in mags
    #dist = 10**(dist_mod/5.0 + 1.0) # distance in parsec
    dist = dist*1000 # convert to parsecs
    metallicity = feh # Metallicity in [M/H]
    mass = 10**5. # Define total cluster mass
    #s_brightness = 30.0 #Desired surface brightness of feature
    #mag_thresh = 23.2 #DECam g magntiude threshold of observations

    ####################################################################################
    #We first need to make an isochrone for this stellar population:
    # Define evolution/atmosphere models and extinction law
    evo_model = evolution.MISTv1()
    atm_func = atmospheres.get_merged_atmosphere
    red_law = reddening.RedLawHosek18b()

    # Specify filters for DECam photometry
    filt_list = ['decam,g', 'decam,r']

    # Make Isochrone object. Note that is calculation will take a few minutes, unless the
    # isochrone has been generated previously.
    my_iso = synthetic.IsochronePhot(logAge, AKs, dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list)


    ####################################################################################
    #Define an IMF and generate the cluster
    # Make multiplicity object
    imf_multi = multiplicity.MultiplicityUnresolved()

    # Make IMF object; we'll use a broken power law with the parameters from Kroupa+01.
    # NOTE: when defining the power law slope for each segment of the IMF, we define
    # the entire exponent, including the negative sign. For example, if dN/dm $\propto$ m^-alpha,
    # then you would use the value "-2.3" to specify an IMF with alpha = 2.3.

    massLimits = np.array([0.2, 0.5, 1, 120]) # Define boundaries of each mass segement
    powers = np.array([-1.3, -2.3, -2.3]) # Power law slope associated with each mass segment
    my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)

    # Make cluster object
    cluster = synthetic.ResolvedCluster(my_iso, my_imf, mass)
    clust = cluster.star_systems
    iso = my_iso.points

    ####################################################################################
    #Generate the mock cluster
    cluster = synthetic.ResolvedCluster(my_iso, my_imf, mass)
    clust = cluster.star_systems

    #Clust is the table of stellar systems in the cluster. It has the following length + columns:
    print(len(clust))
    print(clust.keys())
    
    return cluster

if __name__ == '__main__':
    synthpop(30)
