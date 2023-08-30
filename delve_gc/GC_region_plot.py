#!/usr/bin/env python
# coding: utf-8

###############################
## INSTRUCTIONS TO RUN AS IS:
## python GC_region_plot.py ra dec age [Fe/H] dist_mod r_c r_t False
## The distance modulus is only for visual aid
## Create a folder called 'data', one called 'images', and one called 'gifs'
## Make sure you have downloaded all isochrone files and they are named (for example):
##     'deccam_fem201_124Gyr_ap4.txt' for feh=-2.01 dex, age=12.4 Gyr
## If the DELVE data is already loaded in the form 'GC_data_ra{}_dec{}.fits'
##   with ra and dec rounded to the hundredths, ('ra' and 'dec' are not place
##   holders above, they are there in addition to the numbers), then you can
##   save time by adding 'False' to the end of the call.

## NOTE:
## Currently the background subtraction is not working properly due to the small area
## Working on a fix for this
###############################

# std lib
import warnings
warnings.filterwarnings('ignore') # to suppress some astropy deprecation warnings
import glob
from PIL import Image
import sys

# 3rd party
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, join
import astropy.units as u
import pylab as plt
from scipy.interpolate import interp1d
import movies as mv
import healpy as hp
import scipy.ndimage as nd
from numpy.polynomial import polynomial
import read_mist_models
from matplotlib.patches import Ellipse
import random


# Data Lab
from dl import queryClient as qc
from astroquery.gaia import Gaia

# plots default setup
plt.rcParams['font.size'] = 14

# default is that we need to retrieve the data
get_data=True

ra, dec, age, feh = float(sys.argv[1]), float(sys.argv[2]),float(sys.argv[3]), float(sys.argv[4])
dist_mod = float(sys.argv[5])
r_c = float(sys.argv[6])*u.arcmin
r_t = float(sys.argv[7])*u.arcmin
get_data = eval(sys.argv[8])

# #Pull data for NGC 5897, Age: 12.4 Gyr, [Fe/H] = -2.01 (Koch & Williams 2014)
GC_coord = SkyCoord(ra = ra * u.degree, dec = dec*u.degree)

if get_data:
    print('Selecting region around (ra, dec) = ({}, {})...'.format(ra, dec))

    # # Create the query string; SQL keyword capitalized for clarity
    query ="""SELECT ra,dec,mag_auto_g,mag_auto_r,mag_auto_i,
                        extended_class_g,extended_class_r,extended_class_i,
                        flags_g,flags_r,flags_i,extinction_g,extinction_r,extinction_i,
                        magerr_auto_g,magerr_auto_r,magerr_auto_i, spread_model_g, spreaderr_model_g,quick_object_id
        FROM delve_dr2.objects
        WHERE q3c_radial_query(ra, dec, """ + str(ra) + """, """ + str(dec) + """, 2) AND
              mag_auto_g BETWEEN 14 AND 25 AND
              (mag_auto_g - mag_auto_r) BETWEEN -0.4 AND 1.0"""

    # Get the DELVE data
    R = qc.query(sql=query,fmt='pandas') # R is a pandas DataFrame
    tab = Table.from_pandas(R)
    
    # # Create the query string; SQL keyword capitalized for clarity
    query ="""SELECT ra,dec,bp_g,bp_rp,phot_g_mean_mag,parallax,parallax_error,
                        pm,pmdec,pmdec_error,pmra,pmra_error,source_id,r_med_geo
        FROM gaiaedr3.gaia_source
        JOIN external.gaiaedr3_distance USING (source_id)
        WHERE {}<ra AND {}>ra AND {}<dec AND {}>dec AND
              phot_g_mean_mag BETWEEN 12 AND 25""".format(ra-2, ra+2, dec-2, dec+2)

    print("getting gaia data")
    # Get the Gaia data
    job = Gaia.launch_job_async(query)
    R = job.get_results()
    tab_gaia=R
    
    #Download table that has cross-matched the Gaia and Delve source IDs
    query = """SELECT id1,id2,distance
        FROM delve_dr2.x1p5__objects__gaia_edr3__gaia_source
                        
        WHERE q3c_radial_query(ra1, dec1, """ + str(ra) + """, """ + str(dec) + """, 2)""".format(ra, dec)
    
    R = qc.query(sql=query,fmt='pandas') # R is a pandas DataFrame
    tab_xmatch = Table.from_pandas(R)
    tab_xmatch.rename_column('id1', 'quick_object_id')
    tab_xmatch.rename_column('id2', 'source_id')
    
    tab_xmatch = join(tab, tab_xmatch, join_type='left', keys = 'quick_object_id')
    tab_xmatch = join(tab_xmatch.filled(), tab_gaia, join_type='left', keys = 'source_id')
    
    #Reset ra and dec key value
    tab_xmatch.rename_column('ra_1', 'ra')
    tab_xmatch.rename_column('dec_1', 'dec')
    tab_xmatch.rename_column('r_med_geo', 'gaia_dist')
    
    print('Saving the data to table...')
    tab_xmatch.write('data/GC_data_ra{}_dec{}.fits'.format(np.around(ra, 2), np.around(dec,2)), overwrite=True)

#Load table and de-redden data
print('Loading and Reading Table and Isochrone...')
R = Table.read('data/GC_data_ra{}_dec{}.fits'.format(np.around(ra, 2), np.around(dec,2)))

R['gmag'] = R['mag_auto_g'] - R['extinction_g']
R['rmag'] = R['mag_auto_r'] - R['extinction_r']
R['imag'] = R['mag_auto_i'] - R['extinction_i']

#Load in Dartmouth isochrone:
#iso_file = 'deccam_fem{}_{}Gyr_ap4.txt'.format(int(np.around(-feh*100)), int(np.around(10*age)))
#iso = np.loadtxt(iso_file)

#Load in MIST isochrone:
iso_fehs = np.array([-4.00, -3.50, -3.00, -2.50, -2.00, -1.75, -1.50, -1.25, -1.00, -0.75, -0.50, -0.25])
feh_match = iso_fehs[np.argmin(abs(feh + 0.2 - iso_fehs))]
iso_file = 'MIST_v1.2_vvcrit0.4_DECam/MIST_v1.2_feh_m{:.2f}_afe_p0.0_vvcrit0.4_DECam.iso.cmd'.format(abs(feh_match))
isocmd = read_mist_models.ISOCMD(iso_file)

age_sel = isocmd.age_index(np.log10(age*10**9))
phase = isocmd.isocmds[age_sel]['phase']
phase_sel = (phase == 0) | (phase == 1) | (phase == 2) #Selects MS, RGB section of isochrone
g_iso = isocmd.isocmds[age_sel]['DECam_g'][phase_sel]
r_iso = isocmd.isocmds[age_sel]['DECam_r'][phase_sel]

tolerance = 0.03
iso_low = interp1d(g_iso ,g_iso - r_iso - tolerance, fill_value = 'extrapolate')
iso_high = interp1d(g_iso, g_iso - r_iso + tolerance, fill_value = 'extrapolate')
iso_model = interp1d(g_iso, g_iso - r_iso, fill_value = 'extrapolate')


#Only plot stars within 1*r_h
cat_coords = SkyCoord(ra = R['ra'], dec = R['dec'], unit = (u.deg, u.deg))
seps = cat_coords.separation(GC_coord)

#Quality cut to select stars:
def gc_extclass(spread_model,spreaderr_model):
    sel = np.abs(spread_model) < 0.002 + (3/5.)*spreaderr_model
    return sel
stargal = gc_extclass(R['spread_model_g'], R['spreaderr_model_g'])

qual_cuts_g = (stargal & (R['flags_g'] < 4) & (R['magerr_auto_g'] < 0.5)) & (R['mag_auto_g'] < 23.8)
qual_cuts_r = (stargal & (R['flags_r'] < 4) & (R['magerr_auto_r'] < 0.5)) & (R['mag_auto_r'] < 23.3)
qual_cuts_i = (stargal & (R['flags_i'] < 4) & (R['magerr_auto_i'] < 0.5)) & (R['mag_auto_i'] < 23.0)

#Require no resolved parallax (removes nearby stars)
#parallax_cut = ((R['parallax_over_error'] < 3))

#Require consistent proper motion with central population
pmra_cen, e_pmra_cen, pmdec_cen, e_pmdec_cen, ind_nm_pm = mv.pmExclusion(R, mv.isochrone_selection, [g_iso, r_iso], dist_mod, \
               qual_cuts_g & qual_cuts_i & qual_cuts_r & (seps < 1*r_t), \
               plotfile = 'images/pm_ra{}_dec{}.png'.format(np.around(ra, 2), np.around(dec,2)))

num_nm_pm = len(ind_nm_pm[ind_nm_pm == True])
num_mem_pm = len(ind_nm_pm[ind_nm_pm == False])
ind_no_pm = ~((ind_nm_pm == True) | (ind_nm_pm == False)) #Indices of stars with no pm measurement
ind_no_pm = ind_no_pm.filled(fill_value = True)
ind_nm_pm = ind_nm_pm.filled(fill_value = False) #Indices of stars with pm measurement inconsistent with membership
print('Fraction of PM members: ' + str(num_mem_pm/(num_nm_pm + num_mem_pm)))

#Remove 1-pm_mem_frac of stars with no proper motion measurement.
#This is an approximate way to down-weight stars with no proper motion measurement
#based on their membership probability-- Need to refine this.
ind_remove_nopm = np.array([True for i in range(len(ind_nm_pm))])
inds = np.where(ind_no_pm == True)[0]
inds_exclude = random.sample(list(inds), int(num_nm_pm/(num_nm_pm + num_mem_pm)*(len(inds))))
ind_remove_nopm[inds_exclude] = False

    
#Metallicity cut based on color-color plot(?)

#Distance cut
dist = 10**((dist_mod + 5)/5) # distance in parsecs
#dist_cut0 = np.where(np.abs(R['gaia_dist'] - dist) < 5000)[0]
dist_cut = np.abs(R['gaia_dist'] - dist) < 2000
ind_no_dist = ~((dist_cut == True) | (dist_cut == False))
ind_no_dist = ind_no_dist.filled(fill_value=True) # to include all stars that are within distance or have no distance
# choose to include stars without distances because otherwise there are very few stars
    
#Compile quality cuts
qual_cuts = qual_cuts_g & qual_cuts_r & qual_cuts_i & (~ind_nm_pm) & (~ind_no_dist)# & ind_remove_nopm
print(len(np.where(qual_cuts)[0]))

#Pick 1000 closest stars to GC to generate sample CMD
sample_ind = np.argsort(seps)[:10000]
choose_near = [False for i in range(len(seps))]
count = 0
for i in sample_ind:
    if (qual_cuts[i] == True) & (count < 1000):
        choose_near[i] = True
        count += 1
    
choose_near = choose_near & (~np.isnan(seps)) & qual_cuts

#Select along an isochrone
print('Creating hpxcube...')

#Convert ra/dec to lon/lat offsets from center of GC before running healpy (accounts for projection effect)
offset_coords = SkyCoord(ra = R['ra']*u.deg, dec = R['dec']*u.deg).transform_to(GC_coord.skyoffset_frame())
R['delta_RA'] = offset_coords.lon
R['delta_DEC'] = offset_coords.lat

dist_mods = np.arange(dist_mod-2, dist_mod+2, 0.1)
hpxcube = mv.create_hpxcube(R[qual_cuts], mv.isochrone_selection, [g_iso, r_iso], dist_mods, nside=512)

l_bound, u_bound = mv.getBounds(hpxcube, ra, dec, r_t)

#Save frames
img_names = []
for i in range(len(hpxcube[0])):
    print('Creating image {}'.format(i))
    xmin, xmax = -2.0, +2.0
    ymin, ymax = -2.0, +2.0
    pixscale = 0.03
    sigma=0.06
    nxpix = int((xmax-xmin)/pixscale) + 1
    nypix = int((ymax-ymin)/pixscale) + 1

    x = np.linspace(xmin,xmax,nxpix)
    y = np.linspace(ymin,ymax,nypix)
    XY = np.meshgrid(x,y)
    xx,yy = np.meshgrid(x,y)
    delta_x = x[1] - x[0]
    delta_y = y[1] - y[0]

    hpxmap = hpxcube[:,i]
    nside = hp.get_nside(hpxmap)
    pix = hp.ang2pix(nside,xx.flat,yy.flat,lonlat=True)

    val = hpxmap[pix]
    vv = val.reshape(xx.shape)


    smooth = nd.gaussian_filter(vv, sigma=sigma/pixscale)
    smooth1 = np.copy(smooth)
    
    smooth = smooth.reshape(val.shape)
    smooth1 = smooth1.reshape(val.shape)
    

    #Excise within ~r_t of the GC
    cen_ind = np.sqrt(((np.ravel(xx)))**2 + \
                      (np.ravel(yy))**2) < r_t.value/60.0
    smooth[cen_ind] = np.nan
    
    smooth = smooth.reshape(xx.shape)
    
    #Excise inner region and pixels with  <=1 stars for background model (use 10 stars if not downweighting stars with no PM measurements)
    cen_ind = (np.ravel(xx) > (-0.25)) & ((np.ravel(xx) < (0.25))) & \
                (np.ravel(yy) > (-0.25)) & (np.ravel(yy) < (0.25))
    bkg_smooth = smooth1
    bkg_smooth[cen_ind] = np.nan #np.nanmedian(smooth1[~cen_ind])
    bkg_smooth[vv.reshape(val.shape) <= 10] = np.nan #np.nanmedian(smooth1[~cen_ind])
    deg = 2
    
    xx_bkg = np.copy(xx)
    yy_bkg = np.copy(yy)
    xx_bkg[np.isnan(bkg_smooth).reshape(xx.shape)] = np.nan
    yy_bkg[np.isnan(bkg_smooth).reshape(xx.shape)] = np.nan 
    XY_bkg = [xx_bkg, yy_bkg]
    
    try:
        fit_params, fit_errors = mv.curvefit2d(deg, XY_bkg, np.ravel(bkg_smooth))
        bkg = polynomial.polyval2d(xx, yy, fit_params)
    except:
        bkg = np.zeros(smooth.shape)

    fig, axarr = plt.subplots(2, 2, figsize = (12, 12))#, gridspec_kw={'width_ratios': [[2, 3],[3, 3]]})

    sel = mv.isochrone_selection(R, [g_iso, r_iso], distance_modulus = dist_mods[i])
    axarr[0,0].plot((R['gmag'] - R['rmag'])[choose_near], R['gmag'][choose_near], '.r', zorder =0)
    axarr[0,0].plot((R['gmag'] - R['rmag'])[choose_near & sel], R['gmag'][choose_near & sel], '.b', zorder =1)
    yrange = np.arange(16, 25, 0.1)
    axarr[0,0].plot(iso_model(yrange), yrange + dist_mods[i], '--b', linewidth=3, label = 'DM = {}'.format(dist_mod), zorder=2)
    axarr[0,0].set_xlabel('g-r')
    axarr[0,0].set_ylabel('g')
    axarr[0,0].set_xlim([-0.50, 1])
    axarr[0,0].set_ylim([14, 23])
    axarr[0,0].invert_yaxis()
    
    tidal_circle = Ellipse((0, 0), width=2*r_t.value/60, height=2*r_t.value/60, color='red', fill=False, zorder=2, ec='r', lw=2)
    im = axarr[0,1].pcolormesh(xx,yy,smooth, cmap = 'gray_r',
                             vmin = l_bound, \
                             vmax = 2.0/3*u_bound, zorder=1)
    axarr[0,1].add_patch(tidal_circle)
    axarr[0,1].set_xlim([xmin, xmax])
    axarr[0,1].set_ylim([ymin, ymax])
    axarr[0,1].set_xlabel(r'$\Delta$ RA cos(DEC) (deg)')
    axarr[0,1].set_ylabel(r'$\Delta$ DEC (deg)')
    axarr[0,1].set_title('Distance Modulus: {}'.format(np.around(dist_mods[i], 1)))
    fig.colorbar(im, ax=axarr[0,1])
    
    tidal_circle = Ellipse((0, 0), width=2*r_t.value/60, height=2*r_t.value/60, color='red', fill=False, zorder=2, ec='r', lw=2)
    im2 = axarr[1,1].pcolormesh(xx,yy,smooth-bkg, cmap = 'gray_r', \
                                vmin = l_bound-u_bound/3, \
                                vmax = 1*u_bound/3.0, \
                                zorder=1)
    axarr[1,1].add_patch(tidal_circle)
    axarr[1,1].set_xlim([xmin, xmax])
    axarr[1,1].set_ylim([ymin, ymax])
    axarr[1,1].set_xlabel(r'$\Delta$ RA cos(DEC) (deg)')
    axarr[1,1].set_ylabel(r'$\Delta$ DEC (deg)')
    axarr[1,1].set_title('Bkg Subtracted')
    fig.colorbar(im2, ax=axarr[1,1])
    
    im2 = axarr[1,0].pcolormesh(xx,yy,bkg, cmap = 'gray_r',
                              vmax = 1.5*np.nanmedian(bkg))
    axarr[1,0].set_xlim([xmin, xmax])
    axarr[1,0].set_ylim([ymin, ymax])
    axarr[1,0].set_xlabel(r'$\Delta$ RA cos(DEC) (deg)')
    axarr[1,0].set_ylabel(r'$\Delta$ DEC (deg)')
    axarr[1,0].set_title('Bkg model')
    fig.colorbar(im2, ax=axarr[1,0])
    
    plt.savefig('images/image_ra{}_dec{}_{}.png'.format(np.around(ra, 2),
                                                        np.around(dec,2),np.around(dist_mods[i],1)))

    plt.close()
    img_names.append('image' + str(i) + '.png')
    
    #If at the distance modulus of the system, write out info file which contains:
    #ra, dec, distance, metallicity, foreground count (bkg subtracted and un-subtracted)

    if np.round(dist_mods[i],2) == np.round(dist_mod,2):
        info = open('data/info_ra{}_dec{}.txt'.format(np.around(ra, 2), np.around(dec,2)), 'w')
        
        #Before computing the foreground count per sq. arcsec, find out what fraction of the 
        #foreground estimation area is NaN in the healpix map and account for that incompleteness
        seps_pix = SkyCoord(ra = np.ravel(xx)*u.deg, dec = np.ravel(yy)*u.deg).separation(SkyCoord(0*u.deg, 0*u.deg))
        ind_bkg_nan = np.isnan(np.ravel(vv)) | (np.ravel(vv) == 0)
                        
        frac_nan = len(np.ravel(vv)[(seps_pix > 0.25*u.deg) & (seps_pix < 1.5*u.deg) & ind_bkg_nan])*1.0/len(np.ravel(vv)[(seps_pix > 0.25*u.deg) & (seps_pix < 1.5*u.deg)])
        
        info.write(str(ra) + ' ' + str(dec) + ' ' + str(dist_mod) + ' ' + \
                   str(feh) + ' ' + str(len(R[(seps > 1*u.deg) & (seps < 1.5*u.deg) & \
                                          (sel)])/((1.5**2 - 1.0**2)*(np.pi*3600**2)*(1-frac_nan))))
        info.close()

frames = []
imgs = glob.glob('images/image_ra{}_dec{}*.png'.format(np.around(ra, 2),
                                                        np.around(dec,2),np.around(dist_mods[i],1)))
for i in range(len(imgs)):
    new_frame = Image.open('images/image_ra{}_dec{}_{}.png'.format(np.around(ra, 2),
                                                        np.around(dec,2),np.around(dist_mods[i],1)))
    frames.append(new_frame)
    
frames[0].save('gifs/movie_ra{}_dec{}.gif'.format(np.around(ra, 2), np.around(dec,2)), format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)
