"""
plot_lv.py
Trey V. Wenger - October 2020

Create a longitude-velocity diagram of all knowm HII regions.
"""

import sqlite3
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.coordinates as apycoords
import astropy.units as u

R0 = apycoords.Galactocentric().galcen_distance #distance from sun to galactic center
z0 = apycoords.Galactocentric().z_sun #height of sun above midplane
r0 = np.sqrt(R0**2-z0**2) # distance from sun to galactic center projected along midplane

def convert_galcen_radius_to_gal_radius(glong,glat,Rgal):
    b = -r0*np.cos(glong*np.pi/180)*np.cos(glat*np.pi/180)+z0*np.sin(glat*np.pi/180)
    
    return np.sqrt(b**2-R0**2+Rgal**2) - b
 
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
        SELECT cat.gname,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far,dis.near FROM Detections det
        INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
        INNER JOIN Catalog cat on catdet.catalog_id = cat.id
        INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id 
        WHERE det.vlsr IS NOT NULL AND det.source = "WISE Catalog"
        AND NOT INSTR(det.author, "Anderson") AND NOT INSTR(det.author, "Brown") AND NOT INSTR(det.author, "Wenger")
        AND dis.Rgal IS NOT NULL
        GROUP BY cat.gname, det.component
        ''')
        prehrds = np.array(cur.fetchall(),
                           dtype=[('gname', 'U15'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'), ('far','f8'),('near','f8')])
        print("{0} Pre-HRDS Detections".format(len(prehrds)))
        print(
            "{0} Pre-HRDS Detections with unique GName".format(len(np.unique(prehrds['gname']))))
        print()
        #
        # Get HII regions discovered by HRDS
        #
        cur.execute('''
        SELECT cat.gname,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far,dis.near FROM Detections det
        INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
        INNER JOIN Catalog cat on catdet.catalog_id = cat.id
        INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id
        WHERE det.vlsr IS NOT NULL AND det.source = 'WISE Catalog' AND INSTR(det.author, "Anderson")
        AND dis.Rgal IS NOT NULL
        GROUP BY cat.gname, det.component
        ''')
        hrds = np.array(cur.fetchall(),
                        dtype=[('gname', 'U15'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'), ('far','f8'),('near','f8')])
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
        SELECT cat.gname,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far,dis.near FROM Detections det 
        INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
        INNER JOIN Catalog cat on catdet.catalog_id = cat.id
        INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id
        WHERE det.vlsr IS NOT NULL AND 
        ((det.source="SHRDS Full Catalog" AND det.lines="H88-H112") OR (det.source="SHRDS Pilot" AND det.lines="HS"))
        AND dis.Rgal IS NOT NULL
        GROUP BY cat.gname, det.component HAVING MAX(det.line_snr) ORDER BY cat.gname
        ''')
        shrds_full = np.array(cur.fetchall(),
                              dtype=[('gname', 'U15'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'),('far','f8'),('near','f8')])
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
    prehrds_glat = np.cos(np.deg2rad(prehrds['glat']))
    hrds_glat = np.cos(np.deg2rad(hrds['glat']))
    shrds_glat = np.cos(np.deg2rad(shrds_full['glat']))
    prehrds_x, prehrds_y = prehrds['far']*np.cos(np.deg2rad(prehrds['glong']))-R0.to_value('kpc')*prehrds_glat,       prehrds['far']*np.sin(np.deg2rad(prehrds['glong']))*prehrds_glat
    hrds_x, hrds_y       = hrds['far']*np.cos(np.deg2rad(hrds['glong']))-R0.to_value('kpc')*hrds_glat,                hrds['far']*np.sin(np.deg2rad(hrds['glong']))*hrds_glat
    shrds_x, shrds_y     = shrds_full['far']*np.cos(np.deg2rad(shrds_full['glong']))-R0.to_value('kpc')*shrds_glat,   shrds_full['far']*np.sin(np.deg2rad(shrds_full['glong']))*shrds_glat
    
    # Plot HII region data
    fig2 = plt.figure(figsize=(10,10))
    ax2 = fig2.add_subplot(111)
    
    ax2.plot(prehrds_x*u.kpc, prehrds_y*u.kpc,'ko',markersize=2,label="pre-HRDS")
    ax2.plot(hrds_x*u.kpc,    hrds_y*u.kpc,'md',markersize=2,label="HRDS")
    ax2.plot(shrds_x*u.kpc,   shrds_y*u.kpc,'gs',markersize=2,label="SHRDS")
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig('plots/spiral_allhrds_FAR.pdf')
    plt.close(fig2)
    
    prehrds_x2, prehrds_y2 = prehrds['near']*np.cos(np.deg2rad(prehrds['glong']))-R0.to_value('kpc')*prehrds_glat,        prehrds['near']*np.sin(np.deg2rad(prehrds['glong']))*prehrds_glat
    hrds_x2, hrds_y2       = hrds['near']*np.cos(np.deg2rad(hrds['glong']))-R0.to_value('kpc')*hrds_glat,                 hrds['near']*np.sin(np.deg2rad(hrds['glong']))*hrds_glat
    shrds_x2, shrds_y2     = shrds_full['near']*np.cos(np.deg2rad(shrds_full['glong']))-R0.to_value('kpc')*shrds_glat,    shrds_full['near']*np.sin(np.deg2rad(shrds_full['glong']))*shrds_glat
    
    # Plot HII region data
    fig3 = plt.figure(figsize=(10,10))
    ax3 = fig3.add_subplot(111)
    
    ax3.plot(prehrds_x2*u.kpc,prehrds_y2*u.kpc,'ko',markersize=2,label='pre-HRDS')
    ax3.plot(hrds_x2*u.kpc,   hrds_y2*u.kpc,'md',markersize=2,label='HRDS')
    ax3.plot(shrds_x2*u.kpc,  shrds_y2*u.kpc,'gs',markersize=2,label='SHRDS')
    
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.xlabel("x")
    plt.legend()
    plt.ylabel("y")
    plt.savefig('plots/spiral_allhrds_NEAR.pdf')
    plt.close(fig3)
    
    
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
