#!/usr/bin/env python
"""
Tools for isochrone selection and mapping
"""
__author__ = "Alex Drlica-Wagner"

import numpy as np
import pylab as plt
import healpy as hp
import matplotlib.colors as mcolors
import subprocess
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from numpy.polynomial import polynomial
from astropy.modeling import models, fitting


def angsep(lon1,lat1,lon2,lat2):
    """
    Angular separation (deg) between two sky coordinates.
    Borrowed from astropy (www.astropy.org)
    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1],
    which is slighly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.
    [1] http://en.wikipedia.org/wiki/Great-circle_distance
    """
    lon1,lat1 = np.radians([lon1,lat1])
    lon2,lat2 = np.radians([lon2,lat2])
    
    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.degrees(np.arctan2(np.hypot(num1,num2), denominator))

def box_selection(data, distance_modulus=17.0):
    """Simple isochrone selection based on box in color-magnitude
    Parameters
    ----------
    data : catalog data
    distance_modulus : distance modulus of the box
    Returns
    -------
    sel : binary selection for each object in the catalog
    """
    # This should be moved upstream...
    ra,dec = data['ra'],data['dec'] 
    mag  = data['gmag']
    color = data['gmag'] - data['rmag']

    abs_mag = mag - distance_modulus
    # Define the selection
    sel  = (color < -0.5) & (color > 0.2)
    sel &= (abs_mag > 3.0) & (abs_mag < 5.0)
    return sel

def isochrone_selection(data, isochrone, distance_modulus=17.0):
    """Simple isochrone selection based on a given Dartmouth isochrone
    Parameters
    ----------
    data : catalog data
    isochrone: isochrone txt file
    distance_modulus : distance modulus of the box
    Returns
    -------
    sel : binary selection for each object in the catalog
    """
    
    tolerance = 0.05
    g_iso = isochrone[0]
    r_iso = isochrone[1]

    #Interpolate along isochrone
    iso_low = interp1d(g_iso+distance_modulus ,g_iso - r_iso - tolerance, fill_value = 'extrapolate')
    iso_high = interp1d(g_iso+distance_modulus, g_iso - r_iso + tolerance, fill_value = 'extrapolate')
    iso_model = interp1d(g_iso+distance_modulus, g_iso - r_iso, fill_value = 'extrapolate')
    
    #Select stars within 2sigma of the color tolerance threshold
    sel = (iso_low(data['gmag']) < ((data['gmag'] - data['rmag']) + \
                                     2*np.sqrt(data['magerr_auto_g']**2 + data['magerr_auto_r']**2))) & \
                        (iso_high(data['gmag']) > (data['gmag'] - data['rmag']) - \
                                     2*np.sqrt(data['magerr_auto_g']**2 + data['magerr_auto_r']**2))
                            
               
    return sel

def create_hpxmap(data,selfn,iso,distance_modulus,nside=512):
    """Apply selection and bin data into healpix.
    Parameters
    ----------
    data  : object catalog
    selfn : selection function
    dmod  : distance modulus
    nside : healpix nside
    
    Returns
    -------
    hpxmap : healpix map of counts in pixels
    """

    # Definition of ra/dec may be moved upstream...
    ra,dec = data['delta_RA'],data['delta_DEC'] 
    # Select objects
    sel = selfn(data,isochrone = iso, distance_modulus = distance_modulus)

    pixels = hp.ang2pix(nside, ra[sel],dec[sel],lonlat=True)
    pix,cts = np.unique(pixels,return_counts=True)
    
    hpxmap = np.zeros(hp.nside2npix(nside))
    hpxmap[pix] = cts
    return hpxmap

def create_hpxcube(data,selfn,iso='deccam_fem201_124Gyr_ap4.txt',
                   distance_modulus=np.arange(13, 20, 0.2),nside=512):
    """Create a healpix cube of objects in pixels as a function of
    distance modulus.
    Parameters
    ----------
    data  : catalog data
    selfn : selection function
    distance_modulus : list of distance moduli to create maps for
    nside : resolution of healpix maps
    Returns
    -------
    hpxcube : healpix cube with maps at each distance modulus
    """
    hpxcube = np.zeros((hp.nside2npix(nside),len(distance_modulus)))

    for i,dm in enumerate(distance_modulus):
        hpxmap = create_hpxmap(data,selfn,iso,dm,nside=nside)
        hpxcube[:,i] = hpxmap[:]

    return hpxcube

def hollywood(infiles,outfile=None,delay=40, queue='local'):
    """ Create an animated gif from a set of frames.
    Parameters
    ----------
    infiles : input frame files
    outfile : output gif
    delay   : delay between frames
    queue   : submit to non-local queue (FNAL only)
    Returns
    -------
    None
    """
    print("Lights, Camera, Action...")
    infiles = np.atleast_1d(infiles)
    if not len(infiles): 
        msg = "No input files found"
        raise ValueError(msg)
    
    infiles = ' '.join(infiles)
    if not outfile: outfile = infiles[0].replace('.png','.gif')
    cmd='convert -delay %i -quality 100 %s %s'%(delay,infiles,outfile)
    #if queue != 'local':
    #    cmd = 'csub -q %s '%(queue) + cmd
    print(cmd)
    subprocess.check_call(cmd,shell=True)

def polyfit2d(x, y, f, deg, var=True):
    """
    Fit a 2d polynomial.
    Parameters:
    -----------
    x : array of x values
    y : array of y values
    f : array of function return values
    deg : polynomial degree (length-2 list)
    Returns:
    --------
    c : polynomial coefficients
    """
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c, res, rank, s = np.linalg.lstsq(vander, f, rcond=None)
    return c.reshape(deg+1), res

def polyfuncdeg6(xymesh, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, 
                 c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
                c20, c21, c22, c23, c24, c25, c26, c27, c28, c29,
                c30, c31, c32, c33, c34, c35, c36, c37, c38, c39,
                c40, c41, c42, c43, c44, c45, c46, c47, c48):
    '''
    xymesh: the independent variables
    coeffs: the coefficients that will be fitted, should have len 49
    '''
    (x,y) = xymesh
    
    coeffs = [[c0, c1, c2, c3, c4, c5, c6], 
                [c7, c8, c9, c10, c11, c12, c13], 
                [c14, c15, c16, c17, c18, c19, c20], 
                [c21, c22, c23, c24, c25, c26, c27], 
                [c28, c29, c30, c31, c32, c33, c34], 
                [c35, c36, c37, c38, c39, c40, c41],
                [c42, c43, c44, c45, c46, c47, c48]]
    
    func = polynomial.polyval2d(x, y, coeffs)
    return(np.ravel(func))



def polyfuncdeg5(xymesh, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, 
                 c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
                c20, c21, c22, c23, c24, c25, c26, c27, c28, c29,
                c30, c31, c32, c33, c34, c35):
    '''
    xymesh: the independent variables
    coeffs: the coefficients that will be fitted, should have len 36
    '''
    (x,y) = xymesh
    
    coeffs = [[c0, c1, c2, c3, c4, c5], 
                [c6, c7, c8, c9, c10, c11], 
                [c12, c13, c14, c15, c16, c17], 
                [c18, c19, c20, c21, c22, c23], 
                [c24, c25, c26, c27, c28, c29],
                [c30, c31, c32, c33, c34, c35]]
    
    func = polynomial.polyval2d(x, y, coeffs)
    return(np.ravel(func))

def polyfuncdeg4(xymesh, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, 
                 c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
                c20, c21, c22, c23, c24):
    '''
    xymesh: the independent variables
    coeffs: the coefficients that will be fitted, should have len 25
    '''
    (x,y) = xymesh
    
    coeffs = [[c0, c1, c2, c3, c4],
               [c5, c6, c7, c8, c9], 
               [c10, c11,c12, c13, c14], 
               [c15, c16, c17, c18, c19],
               [c20, c21, c22, c23, c24]]
    
    func = polynomial.polyval2d(x, y, coeffs)
    return(np.ravel(func))

def polyfuncdeg3(xymesh, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, 
                 c10, c11, c12, c13, c14, c15):
    '''
    xymesh: the independent variables
    coeffs: the coefficients that will be fitted, should have len 16
    '''
    (x,y) = xymesh
    
    coeffs = [[c0, c1, c2, c3],
               [c4, c5, c6, c7], 
               [c8, c9,c10, c11], 
               [c12, c13, c14, c15]]
    
    func = polynomial.polyval2d(x, y, coeffs)
    return(np.ravel(func))

def polyfuncdeg2(xymesh, c0, c1, c2, c3, c4, c5, c6, c7, c8):
    '''
    xymesh: the independent variables
    coeffs: the coefficients that will be fitted, should have len 36
    '''
    (x,y) = xymesh
    x = np.ravel(x)
    y = np.ravel(y)
    
    exc = ~np.isnan(x) & ~np.isnan(y)
    x = x[exc]
    y = y[exc]

    coeffs = [[c0, c1, c2], [c3, c4, c5], [c6, c7, c8]]
    
    func = polynomial.polyval2d(x, y, coeffs)

    return(np.ravel(func))

def curvefit2d(deg, xymesh, f):
    '''
    func: function with the parameters to fit
    xymesh: x and y value meshgrid to input into func
    f: already flattened array of the data
    '''
    if deg == 2:
        func = polyfuncdeg2
    elif deg == 3:
        func = polyfuncdeg3
    elif deg == 4:
        func = polyfuncdeg4
    elif deg == 5:
        func = polyfuncdeg5
    elif deg == 6:
        func = polyfuncdeg6
    
    f = f[~np.isnan(f)]

    fit_params, cov_mat = curve_fit(func, xymesh, f, check_finite=False)
    fit_errors = np.sqrt(np.diag(cov_mat))
    
    (x, y) = xymesh
    fit_residual = f - func(xymesh, *fit_params).reshape(f.shape)
    fit_Rsquared = 1 - np.var(fit_residual)/np.var(f)
    
    fit_params = fit_params.reshape(deg+1,deg+1)
    fit_errors = fit_errors.reshape(deg+1,deg+1)
    
    #print('Fit R-squared:', fit_Rsquared)
    #print('Fit Coefficients:', fit_params)
    #print('Fit errors:', fit_errors)
    
    return fit_params, fit_errors

def getBounds(hpxcube, ra, dec, r_t):
    '''
    Derive upper and lower bounds on the values in the colormap that will be 
    used when generating stream-search movies
    Parameters
    ----------
    hpxcube : healpix cube (output from create_hpxcube)
    ra : right ascension of center in movie
    dec : declination of center in movie
    r_t : tidal radius of GC
    Returns
    -------
    lower: value of upper bound
    upper: value of lower bound
    '''
    upper = 0
    lower = 0 # This implementation always returns a lower bound of 0
    
    
    for i in range(len(hpxcube[0])):
        
        #Replicate the processing that is used in generating the movie
        xmin, xmax = -2, +2
        ymin, ymax = -2, +2
        pixscale = 0.05
        sigma=0.03
        nxpix = int((xmax-xmin)/pixscale) + 1
        nypix = int((ymax-ymin)/pixscale) + 1
    
        x = np.linspace(xmin,xmax,nxpix)
        y = np.linspace(ymin,ymax,nypix)
        xx,yy = np.meshgrid(x,y)
    
        hpxmap = hpxcube[:,i]
        nside = hp.get_nside(hpxmap)
        pix = hp.ang2pix(nside,xx.flat,yy.flat,lonlat=True)
    
        val = hpxmap[pix]
        vv = val.reshape(xx.shape)
        smooth = gaussian_filter(vv, sigma=sigma/pixscale)
        smooth = smooth.reshape(val.shape)
        cen_ind = np.sqrt(((np.ravel(xx)))**2 + \
                          (np.ravel(yy))**2) < r_t.value/60.0
        smooth[cen_ind] = np.nan        
        smooth = smooth.reshape(xx.shape)
        
        #Take a multiple of the median value as the upper limit
        upper_t = 3*np.nanmedian(smooth[~np.isnan(smooth) & (vv != 0.0)])
        
        if upper_t > upper:
            upper = upper_t
    
    return lower, upper

def pmExclusion(catalog, selfn, iso, dmod, cuts, plotfile = ''):
    '''
    Derive the systemic proper motion of the GC, and identify stars with 
    proper motions that are inconsistent with this value
    Parameters
    ----------
    catalog : catalog of stars, with proper motion information
    selfn : selection function
    iso   : isochrone
    dmod  : distance modulus
    cuts  : indices of initial quality cuts
    plotfile : location to write diagnostic plot
    Returns
    -------
    pmra_cen: systemic proper motion in the ra direction of the system
    e_pmra_cen: systemic proper motion in the ra direction of the system
    pmdec_cen: systemic proper motion in the dec direction of the system
    e_pmdec_cen: systemic proper motion in the dec direction of the system
    ind_nm: Indices of stars with proper motions inconsistent with the 
            proper motion
    '''
    
    # Select objects along isochrone
    sel = selfn(catalog, isochrone = iso, distance_modulus = dmod)
    
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    #plt.errorbar(catalog['pmra'][sel], catalog['pmdec'][sel], \
    #             xerr = catalog['pmra_error'][sel], \
    #             yerr = catalog['pmdec_error'][sel], fmt='o')
    ax1.set_xlim([-25, 25])
    ax1.set_ylim([-25, 25])
    ax1.set_xlabel('pm_ra', fontsize = 20)
    ax1.set_ylabel('pm_dec', fontsize = 20)
    
    hist, x_edges, y_edges, image = ax1.hist2d(catalog['pmra'][sel & cuts], catalog['pmdec'][sel & cuts], \
                                             bins = np.arange(-25, 25, 0.5), \
                                             norm = mcolors.LogNorm())
        
    # Select objects
    sel = selfn(catalog, isochrone = iso, distance_modulus = dmod)

    #Find overdensity
    ind = np.unravel_index(hist.argmax(), hist.shape)
    
    ax2.set_xlabel('pm_ra', fontsize = 20)
    ax2.set_ylabel('pm_dec', fontsize = 20)
    
    #Fit 2d gaussian to a zoomed in histogram
    hist_zoom = np.transpose(hist[(ind[0]-6):(ind[0]+7), (ind[1]-6):(ind[1]+7)])
    g_init = models.Gaussian2D(amplitude=1., x_mean=x_edges[ind[0]], y_mean=y_edges[ind[1]], x_stddev = 0.5, y_stddev = 0.5)
    fit_g = fitting.LevMarLSQFitter()
    x,y = np.meshgrid(x_edges[(ind[0]-6):(ind[0]+7)], y_edges[(ind[1]-6):(ind[1]+7)])
    g = fit_g(g_init, x, y, hist_zoom)
    
    pmra_cen = g.x_mean + 0.25
    pmdec_cen = g.y_mean + 0.25
    e_pmra = 0.0
    e_pmdec = 0.0

    #Select likely members & re-derive pmra_center and pmdec_center
    ind_mems = np.sqrt((abs(catalog['pmra'] - pmra_cen)/(np.sqrt(catalog['pmra_error']**2 + e_pmra**2)))**2  \
             + (abs(catalog['pmdec'] - pmdec_cen)/(np.sqrt(catalog['pmdec_error'])**2 + e_pmdec**2))**2) < 3
        
    pmra_cen = np.average(catalog['pmra'][ind_mems & cuts & sel], weights = 1/(catalog['pmra_error'][ind_mems & cuts & sel]**2))
    e_pmra_cen = np.sqrt(1/np.sum(1/catalog['pmra_error'][ind_mems & cuts]**2))
    pmdec_cen = np.average(catalog['pmdec'][ind_mems & cuts & sel], weights = 1/(catalog['pmdec_error'][ind_mems & cuts & sel]**2))
    e_pmdec_cen = np.sqrt(1/np.sum(1/catalog['pmdec_error'][ind_mems & cuts & sel]**2))
    
    #Select likely non-members from updated systemic pma and pmdec
    ind_nm = np.sqrt((abs(catalog['pmra'] - pmra_cen)/(np.sqrt(catalog['pmra_error']**2 + e_pmra**2)))**2  \
             + (abs(catalog['pmdec'] - pmdec_cen)/(np.sqrt(catalog['pmdec_error'])**2 + e_pmdec**2))**2) > 3
    
    #plot non-members
    ax2.errorbar(catalog['pmra'][sel & cuts], catalog['pmdec'][sel & cuts], \
                 xerr = catalog['pmra_error'][sel & cuts], \
                 yerr = catalog['pmdec_error'][sel & cuts], fmt='ok')
                  
    #plot members
    ax2.errorbar(catalog['pmra'][sel & ~ind_nm & cuts], catalog['pmdec'][sel & ~ind_nm & cuts], \
                 xerr = catalog['pmra_error'][sel & ~ind_nm & cuts], \
                 yerr = catalog['pmdec_error'][sel & ~ind_nm & cuts], fmt='ob')
        
    #plot centroid pmra/pmdec
    ax2.errorbar(pmra_cen, pmdec_cen, \
                 xerr = e_pmra_cen, \
                 yerr = e_pmdec_cen, fmt='xr', markersize = 10)
        
    ax2.legend(['non-members', 'members', 'pm centroid'])
    
    plt.savefig(plotfile)
    plt.close()
    
    return pmra_cen, e_pmra_cen, pmdec_cen, e_pmdec_cen, ind_nm
