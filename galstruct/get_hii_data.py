#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_hii_data.py

Parse HII region database and extract HII region data.

Copyright(C) 2025 by Trey Wenger <tvwenger@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Trey Wenger - January 2025
"""

import sqlite3
import pandas as pd


def main(db, outfile):
    with sqlite3.connect(db) as conn:
        data = pd.read_sql_query(
            """
            SELECT cat.gname, cat.glong, cat.glat, det.vlsr
            FROM Catalog cat
            INNER JOIN CatalogDetections catdet ON catdet.catalog_id = cat.id
            INNER JOIN Detections det ON catdet.detection_id = det.id
            WHERE det.vlsr IS NOT NULL AND NOT INSTR(det.vlsr, ";")
            AND det.source = "WISE Catalog"
            AND cat.glong > 2.0 AND cat.glong < 358.0
                                 """,
            conn,
        )
    data.to_csv(outfile, index=False)
