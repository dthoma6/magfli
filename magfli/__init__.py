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
from .trace_stop_funcs import trace_stop_earth, \
    trace_stop_box, trace_stop_earth_box, trace_stop_none
from .multitrace import multitrace_cartesian_function, \
    multitrace_cartesian_regular_grid, \
    multitrace_cartesian_unstructured, \
    multitrace_cartesian_unstructured_swmfio
from .fieldlines import integrate_direction, \
    fieldlines_cartesian_unstructured_BATSRUSfile, \
    fieldlines_cartesian_unstructured_swmfio_BATSRUSfile, \
    fieldlines_cartesian_unstructured_VTKfile
