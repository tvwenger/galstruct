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
        SELECT cat.gname,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far FROM Detections det
        INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
        INNER JOIN Catalog cat on catdet.catalog_id = cat.id
        INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id 
        WHERE det.vlsr IS NOT NULL AND det.source = "WISE Catalog"
        AND NOT INSTR(det.author, "Anderson") AND NOT INSTR(det.author, "Brown") AND NOT INSTR(det.author, "Wenger")
        AND dis.Rgal IS NOT NULL
        GROUP BY cat.gname, det.component
        ''')
        prehrds = np.array(cur.fetchall(),
                           dtype=[('gname', 'U15'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'), ('far','f8')])
        print("{0} Pre-HRDS Detections".format(len(prehrds)))
        print(
            "{0} Pre-HRDS Detections with unique GName".format(len(np.unique(prehrds['gname']))))
        print()
        #
        # Get HII regions discovered by HRDS
        #
        cur.execute('''
        SELECT cat.gname,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far FROM Detections det
        INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
        INNER JOIN Catalog cat on catdet.catalog_id = cat.id
        INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id
        WHERE det.vlsr IS NOT NULL AND det.source = 'WISE Catalog' AND INSTR(det.author, "Anderson")
        AND dis.Rgal IS NOT NULL
        GROUP BY cat.gname, det.component
        ''')
        hrds = np.array(cur.fetchall(),
                        dtype=[('gname', 'U15'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'), ('far','f8')])
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
        SELECT cat.gname,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far FROM Detections det 
        INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
        INNER JOIN Catalog cat on catdet.catalog_id = cat.id
        INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id
        WHERE det.vlsr IS NOT NULL AND 
        ((det.source="SHRDS Full Catalog" AND det.lines="H88-H112") OR (det.source="SHRDS Pilot" AND det.lines="HS"))
        AND dis.Rgal IS NOT NULL
        GROUP BY cat.gname, det.component HAVING MAX(det.line_snr) ORDER BY cat.gname
        ''')
        shrds_full = np.array(cur.fetchall(),
                              dtype=[('gname', 'U15'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'),('far','f8')])
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
    prehrds_d_gal = [ convert_galcen_radius_to_gal_radius(prehrds['glong'][i]*u.degree,prehrds['glat'][i]*u.degree,prehrds['Rgal'][i]*u.kpc) for i in range(prehrds['glong'].size)]
    
    # Make astropy coordinates object with the three galactic coordinates
    prehrds_coords = apycoords.Galactic(l=prehrds['glong']*u.degree,b=prehrds['glat']*u.degree,distance=prehrds['far']*u.kpc)
    
    # Convert galactic coordinates to spherical galactocentric coordinates
    prehrds_galcen= apycoords.cartesian_to_spherical(prehrds_coords.transform_to(apycoords.Galactocentric()).x,
                                                     prehrds_coords.transform_to(apycoords.Galactocentric()).y,
                                                     prehrds_coords.transform_to(apycoords.Galactocentric()).z)
    prehrds_dis,prehrds_lat,prehrds_long = prehrds_galcen
    
    # Same calculation for HRDS
    hrds_d_gal = [ convert_galcen_radius_to_gal_radius(hrds['glong'][i]*u.degree,hrds['glat'][i]*u.degree,hrds['Rgal'][i]*u.kpc) for i in range(hrds['glong'].size)]
    hrds_coords = apycoords.Galactic(l=hrds['glong']*u.degree,b=hrds['glat']*u.degree,distance=hrds['far']*u.kpc)
    hrds_galcen= apycoords.cartesian_to_spherical(hrds_coords.transform_to(apycoords.Galactocentric()).x,
                                                  hrds_coords.transform_to(apycoords.Galactocentric()).y,
                                                  hrds_coords.transform_to(apycoords.Galactocentric()).z)
    hrds_dis,hrds_lat,hrds_long = hrds_galcen
    
    shrds_d_gal = [ convert_galcen_radius_to_gal_radius(shrds_full['glong'][i]*u.degree,shrds_full['glat'][i]*u.degree,shrds_full['Rgal'][i]*u.kpc) for i in range(shrds_full['glong'].size)]
    shrds_coords = apycoords.Galactic(l=shrds_full['glong']*u.degree,b=shrds_full['glat']*u.degree,distance=shrds_full['far']*u.kpc)
    shrds_galcen= apycoords.cartesian_to_spherical(shrds_coords.transform_to(apycoords.Galactocentric()).x,
                                                   shrds_coords.transform_to(apycoords.Galactocentric()).y,
                                                   shrds_coords.transform_to(apycoords.Galactocentric()).z)
    shrds_dis,shrds_lat,shrds_long = shrds_galcen
    
    # Plot HII region data
    fig2 = plt.figure(figsize=(10,10))
    ax2 = fig2.add_subplot(111,projection='polar')
    
    ax2.plot(prehrds_long,prehrds_dis,'ko',markersize=2)
    ax2.plot(hrds_long,hrds_dis,'md',markersize=2)
    ax2.plot(shrds_long,shrds_dis,'gs',markersize=2)
    
    plt.savefig('plots/spiral_allhrds.pdf')
    plt.close(fig2)
    
    
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
