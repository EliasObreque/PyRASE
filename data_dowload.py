# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 13:46:57 2025

@author: mndc5
"""

import os
os.environ['ENABLE_IERS_LOAD'] = 'false'
from pyatmos import read_sw_nrlmsise00
from pyatmos import nrlmsise00
# Read from local file
sw_data = read_sw_nrlmsise00('./core/sw_data/SW-All.csv')

# Use NRLMSISE-00 with local data
atm = nrlmsise00(time, alt, lat, lon, f107a, f107, ap, sw_data=sw_data)