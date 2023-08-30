import numpy as np
import scipy
from scipy.integrate import quad

import matplotlib as mpl
import matplotlib.pyplot as plt

import astropy
import astropy.units as u
from astropy.constants import G, k_B, h, c
from astropy.table import Table
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord

import gala
import gala.dynamics as gd
import gala.potential as gp
from gala.potential import NullPotential
import gala.coordinates as gc
from gala.units import galactic
from gala.dynamics.nbody import DirectNBody

from Synth_pop import synthpop

def selfconsistent_plummer(N, M, b):
    """This function generates a stellar system that is consistent 
    with a Plummer, following (more or less) 
    the steps in Aarseth, Henon, Wielen (1974)

    Try to code this up yourself - if you have time."""
    
    np.random.seed(2023)
    
    x1 = np.random.uniform(0, 1, size=N*2)

    r = b * (x1**(-2/3) -1)**-0.5
    r = r[r<5*u.kpc][0:N]

    xyz = np.random.normal(size=(3,N))
    pos = xyz * r.value / np.linalg.norm(xyz, axis=0) * r.unit

    def v_esc(r):
        r = r if isinstance(r, u.Quantity) else r*u.kpc
        pot_abs = G*M / np.sqrt(r**2+b**2)
        return np.sqrt(2 * pot_abs).to('km/s')

    q = np.sqrt(np.random.beta(1.5,4.5, size=N))
    v = q * v_esc(r)

    v_xyz = np.random.normal(size=(3,N))
    vel = v_xyz * v.value / np.linalg.norm(v_xyz, axis=0)*v.unit

    return pos, vel

def gc_sim(pos, vel, M_gc, b_gc, dt, N):
    
    # Create a MW potential that includes the lmc
    mw_w0 = gd.PhaseSpacePosition(
        np.zeros(3)*u.kpc,
        np.zeros(3)*u.km/u.s)

    lmc_w0 = gd.PhaseSpacePosition(
        [0, 250, 0.] * u.kpc,
        [0, 0, -50]*u.km/u.s)


    mw_pot = gp.MilkyWayPotential()
    lmc_pot = gp.NFWPotential.from_circular_velocity(
        v_c=90 * u.km / u.s, r_s=4 * u.kpc, r_ref=5 * u.kpc, units=galactic
    )

    # make the Plummer Potential that describes the GC
    M_gc, b_gc = M_gc*u.Msun, b_gc*u.pc
    print(b_gc)
    gc_pot = gp.PlummerPotential(m=M_gc, b=b_gc, 
                                 units=galactic)
    gc_w0 = gd.PhaseSpacePosition(pos=pos,
                                 vel=vel)
    
    w0 = gd.combine((mw_w0, lmc_w0, gc_w0))
    
    print('    Getting Initial Positions...')
    nbody0 = gd.DirectNBody(w0, particle_potentials=[mw_pot, lmc_pot, gc_pot])
    init_orbit = nbody0.integrate_orbit(dt=-dt*u.Myr, n_steps=int(8000/dt))
    init_pos_mw = init_orbit.pos[-1][0] ; init_vel_mw = init_orbit.vel[-1][0]
    init_pos_lmc = init_orbit.pos[-1][1] ; init_vel_lmc = init_orbit.vel[-1][1]
    init_pos_gc = init_orbit.pos[-1][2] ; init_vel_gc = init_orbit.vel[-1][2]
    
    init_mw_w = gd.PhaseSpacePosition(pos=init_pos_mw.xyz,
                                      vel=init_vel_mw.d_xyz)
    init_lmc_w = gd.PhaseSpacePosition(pos=init_pos_lmc.xyz,
                                        vel=init_vel_lmc.d_xyz)
    init_gc_w = gd.PhaseSpacePosition(pos=init_pos_gc.xyz,
                                      vel=init_vel_gc.d_xyz)
    
    #create a ball of test particles in a Plummer Potential
    N = int(N)
    gc_test_pos, gc_test_vel = selfconsistent_plummer(N=N, M=M_gc, b=b_gc)

    gc_test_w = gd.PhaseSpacePosition(pos=gc_test_pos + init_pos_gc.xyz[:,np.newaxis],
                                      vel=gc_test_vel + init_vel_gc.d_xyz[:,np.newaxis])
    
    # combine phase space positions and potentials
    w0 = gd.combine((init_mw_w, init_lmc_w, init_gc_w, gc_test_w))
    particle_pot = [list([mw_pot, lmc_pot, gc_pot]) +
                    [NullPotential(units=gc_pot.units)] * N][0]

    #Simulate the GC evolution
    print('    Integrating Orbit...')
    nbody1 = DirectNBody(w0, particle_pot, save_all=True)
    orbits = nbody1.integrate_orbit(dt=dt*u.Myr, n_steps=int(8000/dt))
    
    return orbits
    
def paint_pop(tbl, N):
    dist = tbl['Dist']
    cluster = synthpop(dist = dist)
    clust = cluster.star_systems

    decam_g = np.random.choice(clust['m_decam_g'], size=int(N), replace=False) #apparent g mag
    decam_r = np.random.choice(clust['m_decam_r'], size=int(N), replace=False)

    return decam_g, decam_r

def plot_gc_sim(orbits, tbl, N):
    mass = tbl['Mass']*u.Msun
    dist = tbl['Dist']*u.kpc
    r_J_now = tbl['r_J_now']*u.pc
    
    dist_then = np.linalg.norm(orbits[0,2].xyz)
    a_q_then = np.zeros(3)
    a_q_then[0] = dist_then.to(u.kpc).value
    r_J_then = (dist_then * np.power(mass/(3*mw_pot.mass_enclosed(a_q_then*u.kpc)), 1/3)).to(u.kpc).value

    r_J_ICRS = ((r_J_now/dist)*u.rad).to(u.deg).value

    final_ra = orbits.to_coord_frame(coord.ICRS)[-1][2:].ra
    final_dec = orbits.to_coord_frame(coord.ICRS)[-1][2:].dec

    decam_g, _ = paint_pop(tbl, N)
    
    fig, [[ax1,ax2,ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, figsize=(14, 9)) 
    _ = orbits[:, 2].plot(['x', 'y'], axes=[ax1])
    ax1.set_title('Orbit')
    
    _ = orbits[0, 2:].plot(['x', 'y'], axes=[ax2], s=5, c='k')
    ax2.set_title('Initial Cluster')
    ax2.set_xlim(orbits[0,2].x.value-2*r_J_then, orbits[0,2].x.value+2*r_J_then)
    ax2.set_ylim(orbits[0,2].y.value-2*r_J_then, orbits[0,2].y.value+2*r_J_then)
    
    im3 = ax3.scatter(final_ra[1:], final_dec[1:], c=decam_g, s=5, alpha = 0.3, cmap='viridis')
    ax3.set_title('Final Cluster Zoom ICRS')
    ax3.set_xlabel('RA')
    ax3.set_ylabel('Dec')
    ax3.set_xlim(final_ra[0].value-2, final_ra[0].value+2)
    ax3.set_ylim(final_dec[0].value-2, final_dec[0].value+2)
    
    Jacobi2 = plt.Circle((orbits[0,2].x.value, orbits[0,2].y.value), r_J_then, 
                         fill=False, edgecolor='r', zorder=10, label='Jacobi radius')
    Jacobi3 = plt.Circle((final_ra[0].value, final_dec[0].value), r_J_ICRS, 
                         fill=False, edgecolor='r', zorder=10, label='Jacobi radius')
    Jacobi4 = plt.Circle((orbits[-1,2].x.value, orbits[-1,2].y.value), r_J_now.to(u.kpc).value, 
                         fill=False, edgecolor='r', zorder=10, label='Jacobi radius')
#     Jacobi7 = plt.Circle((orbits[-1,0].x.value, orbits[-1,0].y.value), r_J_now.to(u.kpc).value,
#                         fill=False, edgecolor='r', zorder=10, label='r_J')
#     Jacobi8 = plt.Circle((orbits[-1,0].x.value, orbits[-1,0].z.value), r_J_now.to(u.kpc).value,
#                         fill=False, edgecolor='r', zorder=10, label='r_J')
#     Jacobi9 = plt.Circle((orbits[-1,0].y.value, orbits[-1,0].z.value), r_J_now.to(u.kpc).value,
#                         fill=False, edgecolor='r', zorder=10, label='r_J')
    HM_2 = plt.Circle((orbits[0,2].x.value, orbits[0,2].y.value), tbl['r_hm']/1000,
                        fill=False, edgecolor='b', zorder=10, label='half-mass radius')
    HM_4 = plt.Circle((orbits[-1,2].x.value, orbits[-1,2].y.value), tbl['r_hm']/1000,
                        fill=False, edgecolor='b', zorder=10, label='half-mass radius')
#     HM_7 = plt.Circle((orbits[-1,0].x.value, orbits[-1,0].y.value), tbl['r_hm']/1000,
#                         fill=False, edgecolor='b', zorder=10, label='r_hm')
#     HM_8 = plt.Circle((orbits[-1,0].x.value, orbits[-1,0].z.value), tbl['r_hm']/1000,
#                         fill=False, edgecolor='b', zorder=10, label='r_hm')
#     HM_9 = plt.Circle((orbits[-1,0].y.value, orbits[-1,0].z.value), tbl['r_hm']/1000,
#                         fill=False, edgecolor='b', zorder=10, label='r_hm')
    
    _ = orbits[-1, 2:].plot(axes=[ax4, ax5, ax6], s=20, c='k', alpha=1)
    ax4.add_patch(Jacobi4) ; ax4.add_patch(HM_4)
    ax4.set_title('Final All Stars')
    
#     im5 = ax7.hist2d(orbits.x[-1,:].value, orbits.y[-1,:].value, bins=30, 
#                      norm=mpl.colors.LogNorm(vmin=0.1, vmax=10), cmap='gray_r')
#     ax7.add_patch(Jacobi7) ; ax7.add_patch(HM_7)
#     ax7.set_title('Final All Stars')
    
    ax2.add_patch(Jacobi2) ; ax2.add_patch(HM_2)
    ax3.add_patch(Jacobi3)
    
#     _ = orbits[-1, :].plot(axes=[ax7, ax8, ax9], s=5, c='k')
#     ax7.add_patch(Jacobi7) ; ax7.add_patch(HM_7)
#     ax8.add_patch(Jacobi8) ; ax8.add_patch(HM_8)
#     ax9.add_patch(Jacobi9) ; ax9.add_patch(HM_9)
#     ax7.set_xlim(orbits[-1,0].x.value-2*tbl['r_J_now']/1000, orbits[-1,0].x.value+2*tbl['r_J_now']/1000)
#     ax7.set_ylim(orbits[-1,0].y.value-2*tbl['r_J_now']/1000, orbits[-1,0].y.value+2*tbl['r_J_now']/1000)
#     ax8.set_xlim(orbits[-1,0].x.value-2*tbl['r_J_now']/1000, orbits[-1,0].x.value+2*tbl['r_J_now']/1000)
#     ax8.set_ylim(orbits[-1,0].z.value-2*tbl['r_J_now']/1000, orbits[-1,0].z.value+2*tbl['r_J_now']/1000)
#     ax9.set_xlim(orbits[-1,0].y.value-2*tbl['r_J_now']/1000, orbits[-1,0].y.value+2*tbl['r_J_now']/1000)
#     ax9.set_ylim(orbits[-1,0].z.value-2*tbl['r_J_now']/1000, orbits[-1,0].z.value+2*tbl['r_J_now']/1000)
#     ax8.set_title('Final Cluster Zoom Cartesian')
    
    #ax1.legend() ; ax2.legend() ; ax3.legend()
    ax2.legend() ; ax3.legend() ; ax4.legend() ; #ax5.legend() ; ax6.legend()
    #ax7.legend() ; ax8.legend() ; ax9.legend()
    plt.suptitle(tbl['Name'], fontsize=20)
    fig.colorbar(im3, ax=ax3, label='decam g mag')
    fig.tight_layout()
    plt.savefig('figures/' + 'N' + str(N) + tbl['Name'] + '.png')
    
    
if __name__ == '__main__':
    B_tbl = Table.read('data/Baumgardt+19.fit')
    BH_tbl = Table.read('data/BH_18.fit')
    VB_tbl = Table.read('data/VB_21.fit')
    P_tbl = Table.read('data/Piatti_2019.fit')

    B_tbl['Name'] = B_tbl['Name'].astype('S11')

    BH_tbl.rename_column('Cluster', 'Name')
    BH_tbl['Name'] = BH_tbl['Name'].astype('S11')
    BH_tbl.rename_column('__RV_', 'Vrad')
    BH_tbl.rename_column('e__RV_', 'Vrad_err')
    BH_tbl.rename_column('e_Mass', 'Mass_err')
    BH_tbl.rename_column('rnm', 'r_hm')
    BH_tbl.rename_column('rmlp', 'r_hl')
    del BH_tbl['Note']
    del BH_tbl['n_Dist']

    VB_tbl['Name'] = VB_tbl['Name'].astype('S11')
    VB_tbl.rename_column('Rscale', 'Plum_rs')
    VB_tbl.rename_column('e_pmRA', 'pmRA_err')
    VB_tbl.rename_column('pmDE', 'pmDec')
    VB_tbl.rename_column('e_pmDE', 'pmDec_err')
    VB_tbl.rename_column('RAJ2000', 'RA')
    VB_tbl.rename_column('DEJ2000', 'Dec')
    del VB_tbl['SimbadName']
    del VB_tbl['recno']
    del VB_tbl['OName']

    P_tbl.rename_column('rh_rJa', 'rh_rJ_a')
    P_tbl.rename_column('rh_rJRperi', 'rh_rJ_Rperi')
    P_tbl.rename_column('i', 'inc')
    P_tbl['Name'] = P_tbl['Name'].astype('S11')
    
    full_tbl__ = astropy.table.join(BH_tbl, P_tbl, keys='_RA')
    full_tbl__['Name'] = full_tbl__['Name_1']
    del full_tbl__['Name_1']
    del full_tbl__['Name_2']
    full_tbl_ = astropy.table.join(full_tbl__, B_tbl, keys='Name')
    full_tbl = astropy.table.join(full_tbl_, VB_tbl, keys="Name")

    mw_pot = gp.MilkyWayPotential()
    a_q = np.zeros((3, (len(full_tbl))))
    a_q[0] = full_tbl['Dist'] #semi-major axis
    full_tbl['r_J_now'] = (full_tbl['Dist']*np.power(full_tbl['Mass']/(3*mw_pot.mass_enclosed(a_q*u.kpc)), 1/3)).to(u.pc)
    full_tbl['Plum_rs'] = (full_tbl['Plum_rs'].to(u.rad).value*full_tbl['Dist']).to(u.pc)

    col_subset = ['Name', 'RA', 'Dec', 'Dist',
                  'pmRA', 'pmDec', 'Vrad', 
                  'Mass','Plum_rs', 'rc', 'r_hm', 'r_hl', 'r_J_now',
                  'a', 'Rperi', 'rh_rJ_a', 'rh_rJ_Rperi', 'ecc', 'inc']  # List or tuple
    tbl = full_tbl[col_subset]
    
    c = SkyCoord(ra=tbl['RA'], dec=tbl['Dec'], distance=tbl['Dist'],
             pm_ra_cosdec=tbl['pmRA'], pm_dec=tbl['pmDec'], radial_velocity=tbl['Vrad'])
    init_w = c.transform_to(coord.Galactocentric)

    pos = init_w.data.xyz
    vel = init_w.velocity.d_xyz
    
    gcs = [b'NGC 5897   ', b'NGC 5634   ', b'Pal 5      ', b'NGC 288    ', b'NGC 362    ',  
       b'NGC 1851   ', b'NGC 1904   ', b'NGC 2298   ', b'NGC 2808   ', b'E 3        ', 
       b'NGC 3201   ', b'NGC 4147   ',b'NGC 4372   ', b'NGC 5024   ', b'NGC 5286   ', 
       b'NGC 5694   ', b'Pal 4      ', b'NGC 1261   ', b'NGC 5904   ',b'NGC 5986   ', 
       b'Pal 14     ', b'Rup 106    ', b'NGC 4590   ', b'Pal 13     ', b'NGC 6093   ', 
       b'NGC 6101   ', b'NGC 6171   ',b'IC 4499    ', b'Pyxis      ', b'NGC 6362   ', 
       b'NGC 6397   ', b'NGC 6584   ', b'NGC 6723   ', b'NGC 6809   ', b'NGC 7089   ', 
       b'NGC 7099   ', b'NGC 7492   ']
    
    dt=0.5
    for i in range(len(gcs)):
        idx = np.where(tbl['Name'] == gcs[i])[0][0]
        print(tbl['Name'][idx], tbl['RA'][idx], tbl['Dec'][idx])
        
        # First run to check mass loss
        N_0 = 1e3
        orbits = gc_sim(pos[:,idx], vel[:,idx], M_gc = tbl['Mass'][idx], b_gc = tbl['Plum_rs'][idx], 
                        dt=dt, N=N_0)
        retained_mass = ((orbits[-1][3:].pos - orbits[-1][2].pos).norm().to(u.pc).value < tbl['r_J_now'][idx]).sum() / N_0
        print('Cluster retains {}% of initial mass'.format(retained_mass*100))
        
        # Find total number of stars in initial cluster
        total_N = int(tbl['Mass'][idx]/retained_mass)
        print('Total Initial Stars: {}'.format(total_N))
        
        if total_N < 1e4:
            orbits = gc_sim(pos[:,idx], vel[:,idx], M_gc = tbl['Mass'][idx], b_gc = tbl['Plum_rs'][idx], 
                            dt=dt, N=total_N)
        else:
            iterations = int(np.floor(total_N/1e4))
            individual_orbits=[]
            for j in range(iterations):
                individual_orbits_ = gc_sim(pos[:,idx], vel[:,idx], M_gc = tbl['Mass'][idx], b_gc = tbl['Plum_rs'][idx], 
                                dt=dt, N=1e4)
                individual_orbits.append(individual_orbits_[:,3:])
            orbits = gd.combine(individual_orbits)
        
        # Save the orbits for future use
        orbits.to_hdf5('orbits/' + tbl['Name'] + '.hdf5')
        print('    Plotting...')
        plot_gc_sim(orbits, tbl[idx], N=total_N)
        print('')
