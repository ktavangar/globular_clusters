import os
import glob
import logging
from collections import OrderedDict as odict
import gc

import fitsio
import numpy as np
import healpy as hp
# import pylab as plt
import scipy.ndimage as nd
from matplotlib.path import Path
import matplotlib.pyplot as plt

from multiprocessing import Pool
from multiprocessing import Process, Value, Array
from multiprocessing import sharedctypes
#import cPickle as pickle

from astropy import units as u
from astropy.coordinates import SkyCoord

from utils import load_infiles
from ugali.utils import healpix
from ugali.analysis.isochrone import factory as isochrone_factory

from surveys import surveys
import filter_data
import streamlib
from numpy.lib.recfunctions import append_fields

    survey = args.survey
    print('Filtering %s...' % survey)
    mag = surveys[survey]['mag']
    mag_g = mag % 'G'
    mag_r = mag % 'R'
    #mag_i = mag % 'I'
    #mag_z = mag % 'Z'
    ext = surveys[survey]['ext']
    stargal = surveys[survey]['stargal']
    stargal_cut = surveys[survey]['stargal_cut']
    if ext is not None:
        ext = surveys[survey]['ext']
        ext_g = ext % 'G'
        ext_r = ext % 'R'
        ext_i = ext % 'I'
        # fix this, not relevant for now
    minmag = surveys[survey]['minmag']
    maxmag = surveys[survey]['maxmag']
    # columns = ['RA', 'DEC', mag_g, mag_r, mag_i] # include i eventually, try
    # mp search
    columns = ['RA', 'DEC', mag_g, mag_r]#,mag_i]
    if stargal is not None:
        columns.append(stargal)

    ###################
    dmu = 0.2
    # moduli = np.arange(15, 20 + dmu, dmu)
    moduli = np.arange(surveys[survey]['moduli'][0], surveys[
                       survey]['moduli'][1] + dmu, dmu)
    # moduli = [15, 16]
    print('Moduli: ', moduli)
    # 12.0  # from DES search, compared to 12.5, 0.0001, doesn't make much
    # difference along main sequence
    age = float(args.age)
    z = float(args.metallicity)  # 0.0002

    metal_poor = False
    ###################
    
    dirname = surveys[survey]['data_dir']
    print("Reading catalogs from %s..." % dirname)
    filenames = sorted(glob.glob(dirname + '/*.fits'))[:]

    if survey == 'PS1':
        pix = []
        for f in filenames:
            pix.append(int(f[-10:-5]))
        ang = hp.pix2ang(32, pix, nest=False, lonlat=True)
        c = SkyCoord(ang[0], ang[1], frame='icrs', unit='deg')
        b = c.galactic.b.deg
        BMIN = 20
        idx = np.abs(b) > BMIN
        filenames = np.asarray(filenames)[idx]

    data = load_infiles(filenames, columns=columns, multiproc=16)
    # outfile = '/data/des40.b/data/nshipp/stream_search/%s_pickle.dat' % survey
    # pickle.dump(data, outfile)
    gc.collect()

    # Select magnitude range
    print("Selecting: %.1f < %s < %.1f" % (minmag, mag_g, maxmag))
    # data = data[(data[mag_g] < maxmag) & (data[mag_g] > minmag)]
    a1 = data[mag_g] < maxmag
    a2 = data[mag_g] > minmag
    a1 &= a2
    data = data[a1]
    gc.collect()

    mincolor = 0.
    maxcolor = 1.
    print("Selecting: %.1f < %s < %.1f" % (mincolor, 'g - r', maxcolor))
    a1 = data[mag_g] - data[mag_r] < maxcolor
    a2 = data[mag_g] - data[mag_r] > mincolor
    a1 &= a2
    data = data[a1]
    gc.collect()

    if ext is not None:
        data[mag_g] -= ext_g
        data[mag_r] -= ext_r
        data[mag_i] -= ext_i

    if stargal is not None:
        print('Selecting: %s <= %i' % (stargal, stargal_cut))
        a1 = data[stargal] <= stargal_cut
        data = data[a1]
        gc.collect()

    #if metal_poor:
        #sel = filter_data.select_metal_poor(
            #data[mag_g], data[mag_r], data[mag_i])
        #data = data[sel]
        #gc.collect()

    C = surveys[survey]['C']
    E = surveys[survey]['E']
    err = surveys[survey]['err']

    mod = args.distance_modulus
    print("m-M = %.1f..." % (mod))

    gmin = 3.4 + mod

    sel1 = filter_data.select_isochrone(data[mag_g], data[mag_r], err=err, iso_params=[
        mod, age, z], C=C, E=E, gmin=gmin, survey=survey)

    d = data[sel1]
    
    atlas_ends = [30.7, -33.2, 9.3, -20.9] 
    phoenix_ends = [27.9, -42.7, 20.1, -55.3]  
    
    ends = phoenix_ends
    stream_phi1, stream_phi2, R = streamlib.rotation(ends, d['RA'], d['DEC'])
    d = append_fields(d, 'stream_phi1', stream_phi1, usemask=False)
    d = append_fields(d, 'stream_phi2', stream_phi2, usemask=False)
    
    phi1_min, phi1_max = -9, 7.4 #change for atlas/phoenix
    phi2_min, phi2_max = -4, 4
    bins = [int((phi1_max - phi1_min) / 0.2), int((phi2_max - phi2_min) / 0.05)]
    
    
    d = d[np.where((d['stream_phi1'] > phi1_min) & (d['stream_phi1'] < phi1_max) 
                   & (d['stream_phi2'] > phi2_min) & (d['stream_phi2'] < phi2_max))]
    print(d['stream_phi1'])
    #plt.hist2d(d['stream_phi1'], d['stream_phi2'], bins=bins, cmap='gray_r', cmax=14)
    #plt.show()
    stream_catalog = np.histogram2d(d['stream_phi1'], d['stream_phi2'], bins=bins)
    np.save('phoenix_data_catalogs/phoenix_stream_coord_catalog_paper_final.npy', stream_catalog[0])
