#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 13:19:06 2022

@author: Dean Thomas
"""

import logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

from .dipole import dipole_earth_spherical, dipole_earth_cartesian, \
    dipole_mag_line_spherical, dipole_mag_line_delta_cartesian, \
    dipole_earth_cartesian_regular_grid, dipole_earth_cartesian_unstructured
from .trace import trace_stop_earth, \
    trace_stop_box, trace_stop_earth_box, trace_stop_none, \
    trace_field_line_cartesian_function, \
    trace_field_line_cartesian_regular_grid, \
    trace_field_line_cartesian_unstructured
