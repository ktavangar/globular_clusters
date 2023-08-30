#!/usr/bin/env python
# coding: utf-8

# In[80]:


# std lib
from getpass import getpass
import warnings, os
warnings.filterwarnings('ignore') # to suppress some astropy deprecation warnings

# 3rd party
import numpy as np
from numpy.core.defchararray import startswith, count
from astropy import utils, io, convolution, stats
from astropy.visualization import make_lupton_rgb, simple_norm
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from photutils import find_peaks
from pyvo.dal import sia
import pylab as plt
from astropy.utils.data import download_file
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import interp1d
import movies as mv
#get_ipython().run_line_magic('matplotlib', 'inline')

# Data Lab
#from dl import authClient as ac, queryClient as qc
#from dl.helpers.utils import convert

# plots default setup
#plt.rcParams['font.size'] = 14
#qc.set_profile('delve_private')


# In[45]:


# #Pull data for NGC 5897, Age: 12.4 Gyr, [Fe/H] = -2.01 (Koch & Williams 2014)
NGC5897_coord = SkyCoord(ra = '15h17m24.40s', dec = '-21d00m36.4s', unit = (u.hourangle, u.deg))
# ra = NGC5897_coord.ra.deg
# dec = NGC5897_coord.dec.deg

# print(ra)
# print(dec)

# # Create the query string; SQL keyword capitalized for clarity

# query ="""SELECT ra,dec,mag_auto_g,mag_auto_r,mag_auto_i,extended_class_g,extended_class_r,    extended_class_i,flags_g,flags_r,flags_i,extinction_g,extinction_r,extinction_i,    magerr_auto_g,magerr_auto_r,magerr_auto_i
#    FROM delve_y3t3.objects
#    WHERE q3c_radial_query(ra, dec, """ + str(ra) + """, """ + str(dec) + """, 2) AND
#          mag_auto_g BETWEEN 14 AND 25 AND
#          (mag_auto_g - mag_auto_r) BETWEEN -0.4 AND 1.0""".format(ra, dec)


# In[76]:


# get_ipython().run_line_magic('time', '')
# R = qc.query(sql=query,fmt='pandas') # R is a pandas DataFrame


# In[83]:


# #Write to a data file
# tab = Table.from_pandas(R)
# tab.write('GC_data.fits', overwrite=True)
# command = 'java -jar stilts.jar cdsskymatch cdstable=I/350/gaiaedr3 find=all                     in=GC_data.fits ra=ra dec=dec                     radius=1 out=/home/jail/dlusers/achiti/notebooks/03_ScienceExamples/DwarfGalaxie/GC_data_Gaia.fits'
# os.system(command)


# In[122]:


#Load table and de-redden data
R = Table.read('GC_Gaia_data.fits')

R['gmag'] = R['mag_auto_g'] - R['extinction_g']
R['rmag'] = R['mag_auto_r'] - R['extinction_r']
R['imag'] = R['mag_auto_i'] - R['extinction_i']

plt.hist(R['gmag'][(R['flags_g'] < 4)])


# In[98]:


#Load in Dartmouth isochrone:
iso = np.loadtxt('deccam_fem201_124Gyr_ap4.txt')
dist_mod = 15.55 #Koch & McWilliam 2014
tolerance = 0.03
ind_g = 6
ind_r = 7
ind_i = 8
iso_low = interp1d(iso[:,ind_g]+dist_mod ,iso[:,ind_g] - iso[:,ind_r] - tolerance, fill_value = 'extrapolate')
iso_high = interp1d(iso[:,ind_g]+dist_mod, iso[:,ind_g] - iso[:,ind_r] + tolerance, fill_value = 'extrapolate')
iso_model = interp1d(iso[:,ind_g]+dist_mod, iso[:,ind_g] - iso[:,ind_r], fill_value = 'extrapolate')

ind_iso = (iso_low(R['gmag'].filled()) < ((R['gmag'].filled() - R['rmag'].filled()) +                                  2*np.sqrt(R['magerr_auto_g'].filled()**2 + R['magerr_auto_r'].filled()**2))) &                     (iso_high(R['gmag'].filled()) > (R['gmag'].filled() - R['rmag'].filled()) -                                  2*np.sqrt(R['magerr_auto_g'].filled()**2 + R['magerr_auto_r'].filled()**2))


# In[99]:


#Only plot stars within 1*r_h
cat_coords = SkyCoord(ra = R['ra'], dec = R['dec'], unit = (u.deg, u.deg))
r_h = 2.06*u.arcmin
seps = cat_coords.separation(NGC5897_coord)


# In[114]:


#Quality cut to select stars:
qual_cuts_g = ((R['extended_class_g'] == 0) | (R['extended_class_g'] == 1)) & (R['flags_g'] < 4) & (R['magerr_auto_g'] < 0.5)
qual_cuts_r = ((R['extended_class_r'] == 0) | (R['extended_class_r'] == 1)) & (R['flags_r'] < 4) & (R['magerr_auto_r'] < 0.5)
qual_cuts_i = ((R['extended_class_i'] == 0) | (R['extended_class_i'] == 1)) & (R['flags_i'] < 4) & (R['magerr_auto_i'] < 0.5)

#Require no resolved parallax (removes nearby stars)
#parallax_cut = ((R['parallax_over_error'] < 3))

#Require consistent proper motion with central population


#Metallicity cut based on color-color plot(?)

qual_cuts = qual_cuts_g & qual_cuts_r & qual_cuts_i# & parallax_cut

choose = (seps < 1*r_h) & (~np.isnan(seps)) & qual_cuts
choose2 = (seps < 2*r_h) & (~np.isnan(seps)) & qual_cuts
choose3 = (seps > 4*r_h) & (~np.isnan(seps)) & qual_cuts


#Select along an isochrone
sel = mv.isochrone_selection(R, distance_modulus = dist_mod)
hpxcube = mv.create_hpxcube(R, mv.isochrone_selection, \
                           [dist_mod - 1.0, dist_mod, dist_mod + 1.0])
mv.hollywood(hpxcube, outfile = 'test')#(?)


fig, axarr = plt.subplots(2, 2, figsize = (15, 12))


axarr[0,0].set_title('NGC5897 (< 2*r_h)')
yrange = np.arange(16, 25, 0.1)
axarr[0,0].plot((R['gmag'] - R['rmag'])[choose2 & sel], R['gmag'][choose2 & sel], '.k', zorder =0)
axarr[0,0].plot((R['gmag'] - R['rmag'])[choose2 & sel], R['gmag'][choose2 & sel], '.b', zorder =1)
axarr[0,0].plot(iso_model(yrange), yrange, '--r', linewidth=3)
axarr[0,0].set_xlabel('g-r')
axarr[0,0].set_ylabel('g')
axarr[0,0].set_xlim([-0.50, 1])
axarr[0,0].set_ylim([14, 23])
axarr[0,0].invert_yaxis()

# axarr[1,0].set_title('NGC5897 (> 4*r_h)')
# yrange = np.arange(16, 25, 0.1)
# axarr[1,0].plot((R['gmag'] - R['rmag'])[choose3],                 R['gmag'][choose3], '.k', zorder =0)
# axarr[1,0].plot((R['gmag'] - R['rmag'])[choose3 & ind_iso],                 R['gmag'][choose3 & ind_iso], '.b', zorder =1)
# axarr[1,0].plot(iso_model(yrange), yrange, '--r', linewidth=3)
# axarr[1,0].set_xlabel('g-r')
# axarr[1,0].set_ylabel('g')
# axarr[1,0].set_xlim([-0.50, 1])
# axarr[1,0].set_ylim([14, 23])
# axarr[1,0].invert_yaxis()

# plt.savefig('f3.png')

plt.show()


# In[ ]:


#Implement pixelizer and scanner

