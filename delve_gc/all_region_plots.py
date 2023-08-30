import os

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from astroquery.vizier import Vizier

import pylab as plt
import numpy as np

Vizier.ROW_LIMIT = -1
catalogs = Vizier(columns=['*', 'Rh', 'Rc', 'c']).get_catalogs('VII/202')

tab = catalogs[0]
coords = SkyCoord(tab['RAJ2000'],tab['DEJ2000'],unit = (u.hourangle, u.deg))
print(len(coords))
print(coords)

#Select GCs in DELVE pre-DR2
#sel = (coords.galactic.b.deg > 10) & (coords.dec.deg < 30)

delve_gcs = coords
ras = delve_gcs.ra.value
decs = delve_gcs.dec.value
fehs = tab['__Fe_H_']
dists = 5*np.log10(tab['Rsun']*1000) - 5
rcs = tab['Rc']
rts = 10**(tab['c'])*tab['Rc']

for i in range(len(delve_gcs)):
    os.system("python GC_region_plot.py {} {} 12.4 {} {} {} {} True".format(ras[i], decs[i], fehs[i], dists[i], rcs[i], rts[i]))
