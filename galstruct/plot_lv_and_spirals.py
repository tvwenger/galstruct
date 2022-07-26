"""
plot_lv.py
Trey V. Wenger - October 2020

Create a longitude-velocity diagram of all knowm HII regions. Also create
a face on plot of the galaxy using any known galactocentric locations
"""

import sqlite3
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.coordinates as apycoords
import astropy.units as u
import matplotlib as mpl




R0 = apycoords.Galactocentric().galcen_distance #distance from sun to galactic center
z0 = apycoords.Galactocentric().z_sun #height of sun above midplane
r0 = np.sqrt(R0**2-z0**2) # distance from sun to galactic center projected along midplane

def convert_galcen_radius_to_gal_radius(glong,glat,Rgal):
    b = -r0*np.cos(glong*np.pi/180)*np.cos(glat*np.pi/180)+z0*np.sin(glat*np.pi/180)
    
    return np.sqrt(b**2-R0**2+Rgal**2) - b

@mpl.rc_context({"backend"            : "TkAgg",
                "font.family"         : "DejaVu Sans",
                "font.size"           : 16.0,
                "font.serif"          : "Computer Modern",
                "text.usetex"         : True,
                "text.hinting_factor" : 8 ,
                "mathtext.default"    : "regular", 
                "axes.grid"           : True,   
                "xtick.major.pad"     : 8,
                "ytick.major.pad"     : 8,
                "ps.usedistiller"     : "xpdf"}
                )
def main(db='/data/hii_v1_20200910.db'):
    #
    # Read detections from database
    #
    with sqlite3.connect(db) as conn:
        cur = conn.cursor()
        cur.execute('PRAGMA foreign_keys = ON')
        #
        # Get previously-known HII Regions
        # (i.e. not GBT HRDS and not SHRDS)
        #
        cur.execute('''
        SELECT cat.gname,cat.kdar,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far,dis.near,dis.tangent FROM Detections det
        INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
        INNER JOIN Catalog cat on catdet.catalog_id = cat.id
        INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id 
        WHERE det.vlsr IS NOT NULL AND det.source = "WISE Catalog" AND cat.kdar IS NOT NULL
        AND NOT INSTR(det.author, "Anderson") AND NOT INSTR(det.author, "Brown") AND NOT INSTR(det.author, "Wenger")
        AND dis.Rgal IS NOT NULL
        GROUP BY cat.gname, det.component
        ''')
        prehrds = np.array(cur.fetchall(),
                           dtype=[('gname', 'U15'), ('kdar','U1'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'), ('far','f8'),('near','f8'),('tangent','f8')])
        print("{0} Pre-HRDS Detections".format(len(prehrds)))
        print(
            "{0} Pre-HRDS Detections with unique GName".format(len(np.unique(prehrds['gname']))))
        print()
        #
        # Get HII regions discovered by HRDS
        #
        cur.execute('''
        SELECT cat.gname,cat.kdar,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far,dis.near,dis.tangent FROM Detections det
        INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
        INNER JOIN Catalog cat on catdet.catalog_id = cat.id
        INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id 
        WHERE det.vlsr IS NOT NULL AND det.source = 'WISE Catalog' AND INSTR(det.author, "Anderson")
        AND dis.Rgal IS NOT NULL AND cat.kdar IS NOT NULL
        GROUP BY cat.gname, det.component
        ''')
        hrds = np.array(cur.fetchall(),
                        dtype=[('gname', 'U15'), ('kdar','U1'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'), ('far','f8'),('near','f8'),('tangent','f8')])
        # remove any sources in previously-known
        good = np.array([gname not in prehrds['gname']
                         for gname in hrds['gname']])
        hrds = hrds[good]
        print("{0} HRDS Detections".format(len(hrds)))
        print("{0} HRDS Detections with unique GName".format(
            len(np.unique(hrds['gname']))))
        print()
        #
        # Get HII regions discovered by SHRDS Full Catalog
        # Limit to stacked detection with highest line_snr
        #
        cur.execute('''
        SELECT cat.gname,cat.kdar,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far,dis.near,dis.tangent FROM Detections det 
        INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
        INNER JOIN Catalog cat on catdet.catalog_id = cat.id
        INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id
        WHERE det.vlsr IS NOT NULL AND 
        ((det.source="SHRDS Full Catalog" AND det.lines="H88-H112") OR (det.source="SHRDS Pilot" AND det.lines="HS"))
        AND dis.Rgal IS NOT NULL AND cat.kdar IS NOT NULL
        GROUP BY cat.gname, det.component HAVING MAX(det.line_snr) ORDER BY cat.gname
        ''')
        shrds_full = np.array(cur.fetchall(),
                              dtype=[('gname', 'U15'), ('kdar','U1'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'),('far','f8'),('near','f8'),('tangent','f8')])
        # remove any sources in previously-known or GBT HRDS
        good = np.array([(gname not in prehrds['gname']) and (gname not in hrds['gname'])
                         for gname in shrds_full['gname']])
        shrds_full = shrds_full[good]
        print("{0} SHRDS Full Catalog Detections".format(len(shrds_full)))
        print("{0} SHRDS Full Catalog Detections with unique GName".format(
            len(np.unique(shrds_full['gname']))))
        print()
        print("{0} Total Detections".format(
            len(prehrds)+len(hrds)+len(shrds_full)))
        print("{0} Total Detections with unique GName".format(len(np.unique(
            np.concatenate((prehrds['gname'], hrds['gname'], shrds_full['gname']))))))
        print()
        #
        # Get all WISE Catalog objects
        #
        cur.execute('''
        SELECT cat.gname, cat.catalog  FROM Catalog cat
        ''')
        wise = np.array(cur.fetchall(),
                        dtype=[('gname', 'U15'), ('catalog', 'U1')])
        #
        # Get all continuum detections without RRL detections
        #
        cur.execute('''
        SELECT cat.gname, det.cont, det.vlsr, COALESCE(det.line_snr, 1.0) AS snr FROM Detections det
        INNER JOIN CatalogDetections catdet ON catdet.detection_id = det.id
        INNER JOIN Catalog cat ON catdet.catalog_id = cat.id
        WHERE det.source = 'SHRDS Full Catalog' AND det.lines = 'H88-H112'
        AND cat.catalog = 'Q'
        GROUP BY cat.gname HAVING MAX(snr)
        ''')
        wise_quiet = np.array(cur.fetchall(),
                              dtype=[('gname', 'U15'), ('cont', 'f8'), ('vlsr', 'f8'), ('snr', 'f8')])
        #
        # Count known
        #
        not_quiet = np.sum(np.isnan(wise_quiet['vlsr']))
        print("SHRDS found continuum emission but no RRL emission toward {0} sources".format(
            not_quiet))
        known = np.sum(wise['catalog'] == 'K')+len(set(wise['gname'][wise['catalog'] != 'K']
                                                       ).intersection(np.concatenate((prehrds['gname'], hrds['gname'], shrds_full['gname']))))
        candidate = np.sum(wise['catalog'] == 'C') - len(set(wise['gname'][wise['catalog'] == 'C']).intersection(
            np.concatenate((prehrds['gname'], hrds['gname'], shrds_full['gname'])))) + not_quiet
        quiet = np.sum(wise['catalog'] == 'Q') - len(set(wise['gname'][wise['catalog'] == 'Q']).intersection(
            np.concatenate((prehrds['gname'], hrds['gname'], shrds_full['gname'])))) - not_quiet
        group = np.sum(wise['catalog'] == 'G') - len(set(wise['gname'][wise['catalog'] == 'G']
                                                         ).intersection(np.concatenate((prehrds['gname'], hrds['gname'], shrds_full['gname']))))
        print("Now WISE Catalog containers:")
        print("{0} known".format(known))
        print("{0} candidate".format(candidate))
        print("{0} quiet".format(quiet))
        print("{0} group".format(group))
    #
    # Save to file
    #
    all_data = np.concatenate([prehrds, hrds, shrds_full])
    all_data = sorted(all_data, key=lambda x: x[0])
    with open('all_hii_lv.txt', 'w') as f:
        labels = ['GName', 'GLong', 'GLat', 'VLSR', 'IR_Radius', 'RRL_Author']
        units = ['', 'deg', 'deg', 'km/s', '', 'arcsec', '']
        f.write('{0:16}, {1:6}, {2:6}, {3:7}, {4:9}, {5}\n'.format(*labels))
        f.write('#{0:16}, {1:6}, {2:6}, {3:7}, {4:9}, {5}\n'.format(*units))
        for hii in all_data:
            f.write('{0:16}, {1:6.2f}, {2:6.2f}, {3:7.2f}, {4:9.1f}, {5}\n'.format(
                hii['gname'], hii['glong'], hii['glat'], hii['vlsr'], hii['radius'], hii['author']))
    
    
    # ---------- MAKE SPIRAL PLOT -------------
    # Convert (galactic longitude, galactic latitude, galactocentric distance) to galactic coordinates distance
    prehrds_far = prehrds[prehrds['kdar']=='F']
    hrds_far = hrds[hrds['kdar']=='F']
    shrds_far = shrds_full[shrds_full['kdar']=='F']
    prehrds_glat = np.cos(np.deg2rad(prehrds_far['glat']))
    hrds_glat = np.cos(np.deg2rad(hrds_far['glat']))
    shrds_glat = np.cos(np.deg2rad(shrds_far['glat']))
    prehrds_x, prehrds_y = prehrds_far['far']*np.cos(np.deg2rad(prehrds_far['glong']))-R0.to_value('kpc')*prehrds_glat, prehrds_far['far']*np.sin(np.deg2rad(prehrds_far['glong']))*prehrds_glat
    hrds_x, hrds_y       = hrds_far['far']*np.cos(np.deg2rad(hrds_far['glong']))-R0.to_value('kpc')*hrds_glat,          hrds_far['far']*np.sin(np.deg2rad(hrds_far['glong']))*hrds_glat
    shrds_x, shrds_y     = shrds_far['far']*np.cos(np.deg2rad(shrds_far['glong']))-R0.to_value('kpc')*shrds_glat,       shrds_far['far']*np.sin(np.deg2rad(shrds_far['glong']))*shrds_glat
    
    
    # Plot HII region data
    fig2 = plt.figure(figsize=(10,10))
    ax2 = fig2.add_subplot(111)
    
    ax2.plot(prehrds_y*u.kpc, -prehrds_x*u.kpc,'ko',markersize=2,label="pre-HRDS")
    ax2.plot(hrds_y*u.kpc,    -hrds_x*u.kpc,'md',markersize=2,label="HRDS")
    ax2.plot(shrds_y*u.kpc,   -shrds_x*u.kpc,'gs',markersize=2,label="SHRDS")
    plt.ylim(-20,20)
    plt.xlim(-20,20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.savefig('plots/spiral_allhrds_FAR.pdf')
    plt.close(fig2)
    
    prehrds_near = prehrds[prehrds['kdar']=='N']
    hrds_near = hrds[hrds['kdar']=='N']
    shrds_near = shrds_full[shrds_full['kdar']=='N']
    prehrds_glat = np.cos(np.deg2rad(prehrds_near['glat']))
    hrds_glat = np.cos(np.deg2rad(hrds_near['glat']))
    shrds_glat = np.cos(np.deg2rad(shrds_near['glat']))
    prehrds_x2, prehrds_y2 = prehrds_near['near']*np.cos(np.deg2rad(prehrds_near['glong']))-R0.to_value('kpc')*prehrds_glat, prehrds_near['near']*np.sin(np.deg2rad(prehrds_near['glong']))*prehrds_glat
    hrds_x2, hrds_y2       = hrds_near['near']*np.cos(np.deg2rad(hrds_near['glong']))-R0.to_value('kpc')*hrds_glat,          hrds_near['near']*np.sin(np.deg2rad(hrds_near['glong']))*hrds_glat
    shrds_x2, shrds_y2     = shrds_near['near']*np.cos(np.deg2rad(shrds_near['glong']))-R0.to_value('kpc')*shrds_glat,       shrds_near['near']*np.sin(np.deg2rad(shrds_near['glong']))*shrds_glat
    
    # Plot HII region data
    fig3 = plt.figure(figsize=(10,10))
    ax3 = fig3.add_subplot(111)
    
    ax3.plot(prehrds_y2*u.kpc,-prehrds_x2*u.kpc,'ko',markersize=2,label='pre-HRDS')
    ax3.plot(hrds_y2*u.kpc,   -hrds_x2*u.kpc,'md',markersize=2,label='HRDS')
    ax3.plot(shrds_y2*u.kpc,  -shrds_x2*u.kpc,'gs',markersize=2,label='SHRDS')
    plt.ylim(-20,20)
    plt.xlim(-20,20)
    plt.xlabel("x")
    plt.legend()
    plt.ylabel("y")
    plt.grid()
    plt.savefig('plots/spiral_allhrds_NEAR.pdf')
    plt.close(fig3)
    
    prehrds_tangent = prehrds[prehrds['kdar']=='T']
    hrds_tangent = hrds[hrds['kdar']=='T']
    shrds_tangent = shrds_full[shrds_full['kdar']=='T']
    prehrds_glat = np.cos(np.deg2rad(prehrds_tangent['glat']))
    hrds_glat = np.cos(np.deg2rad(hrds_tangent['glat']))
    shrds_glat = np.cos(np.deg2rad(shrds_tangent['glat']))
    prehrds_x3, prehrds_y3 = prehrds_tangent['tangent']*np.cos(np.deg2rad(prehrds_tangent['glong']))-R0.to_value('kpc')*prehrds_glat, prehrds_tangent['tangent']*np.sin(np.deg2rad(prehrds_tangent['glong']))*prehrds_glat
    hrds_x3, hrds_y3       = hrds_tangent['tangent']*np.cos(np.deg2rad(hrds_tangent['glong']))-R0.to_value('kpc')*hrds_glat,          hrds_tangent['tangent']*np.sin(np.deg2rad(hrds_tangent['glong']))*hrds_glat
    shrds_x3, shrds_y3     = shrds_tangent['tangent']*np.cos(np.deg2rad(shrds_tangent['glong']))-R0.to_value('kpc')*shrds_glat,       shrds_tangent['tangent']*np.sin(np.deg2rad(shrds_tangent['glong']))*shrds_glat
    
    # Plot HII region data
    fig4 = plt.figure(figsize=(10,10))
    ax4 = fig4.add_subplot(111)
    
    ax4.plot(prehrds_y3*u.kpc,-prehrds_x3*u.kpc,'ko',markersize=2,label='pre-HRDS')
    ax4.plot(hrds_y3*u.kpc,   -hrds_x3*u.kpc,'md',markersize=2,label='HRDS')
    ax4.plot(shrds_y3*u.kpc,  -shrds_x3*u.kpc,'gs',markersize=2,label='SHRDS')
    plt.ylim(-20,20)
    plt.xlim(-20,20)
    plt.grid()
    plt.xlabel("x")
    plt.legend()
    plt.ylabel("y")
    plt.savefig('plots/spiral_allhrds_TANGENT.pdf')
    plt.close(fig4)
    
    fig5 = plt.figure(figsize=(10,10))
    ax5=fig5.add_subplot(111)
    
    ax5.plot(prehrds_y*u.kpc,-prehrds_x*u.kpc,'ko',markersize=2,label='pre-HRDS')
    ax5.plot(hrds_y*u.kpc,   -hrds_x*u.kpc,'md',markersize=2,label='HRDS')
    ax5.plot(shrds_y*u.kpc,  -shrds_x*u.kpc,'gs',markersize=2,label='SHRDS')
    ax5.plot(prehrds_y2*u.kpc,-prehrds_x2*u.kpc,'ko',markersize=2)
    ax5.plot(hrds_y2*u.kpc,   -hrds_x2*u.kpc,'md',markersize=2)
    ax5.plot(shrds_y2*u.kpc,  -shrds_x2*u.kpc,'gs',markersize=2)
    ax5.plot(prehrds_y3*u.kpc,-prehrds_x3*u.kpc,'ko',markersize=2)
    ax5.plot(hrds_y3*u.kpc,   -hrds_x3*u.kpc,'md',markersize=2)
    ax5.plot(shrds_y3*u.kpc,  -shrds_x3*u.kpc,'gs',markersize=2)
    
    plt.ylim(-20,20)
    plt.xlim(-20,20)
    plt.xlabel("x")
    plt.grid()
    plt.legend()
    plt.ylabel("y")
    plt.savefig('plots/spiral_allhrds_ALL.pdf')
    plt.close(fig5)
    
    # ------------MAKE LV PLOT -----------------
    #
    # Fix longitudes
    #
    fix = prehrds['glong'] > 180.
    prehrds['glong'][fix] = prehrds['glong'][fix] - 360.
    fix = hrds['glong'] > 180.
    hrds['glong'][fix] = hrds['glong'][fix] - 360.
    fix = shrds_full['glong'] > 180.
    shrds_full['glong'][fix] = shrds_full['glong'][fix] - 360.
    #
    # Plot l-v diagram
    #
    fig = plt.figure(figsize=(8.5, 10))
    ax = fig.add_subplot(111)
    ax.plot(prehrds['vlsr'], prehrds['glong'], 'ko',
            label='Previously Known', markersize=2)
    ax.plot(hrds['vlsr'], hrds['glong'], 'md', label='HRDS', markersize=2)
    ax.plot(shrds_full['vlsr'], shrds_full['glong'],
            'gs', label='SHRDS', markersize=2)
    # Change yticks to positive
    ax.set_yticklabels(['{0}'.format(glong)
                        if glong >= 0 else '{0}'.format(360 + glong)
                        for glong in np.arange(-180, 181, 20)])
    ax.legend(loc='upper right', fontsize=12, markerscale=3)
    ax.set_xlabel(r'$V_{\rm LSR}$ (km s$^{-1}$)')
    ax.set_ylabel(r'Galactic Longitude (deg)')
    ax.set_xlim(-150, 150)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-150, 151, 50))
    ax.set_yticks(np.arange(-180, 181, 20))
    fig.tight_layout()
    plt.savefig('plots/lv_allhrds.pdf')
    plt.close(fig)
    

if __name__ == "__main__":
    main(db="data/hii_v2_20201203.db")
