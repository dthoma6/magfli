#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:33:08 2022

@author: Dean Thomas
"""

import numpy as np
import logging
from ..dipole import dipole_earth_cartesian, dipole_earth_spherical, \
                    dipole_earth_cartesian_regular_grid, \
                    dipole_earth_cartesian_unstructured, \
                    dipole_mag_line_delta_cartesian
from ..trace import trace_field_line_cartesian_function, \
                    trace_field_line_cartesian_regular_grid, \
                    trace_field_line_cartesian_unstructured, \
                    trace_stop_earth

def test_dipole_fields():
    """Compare results from dipole_earth_cartsian and dipole_earth_spherical
 
    
    Inputs
    -------
    None.
    
    Returns
    -------
    None.
    """
    
    # Point where we will evaluate the magnetic fields
    x = 1
    y = 3
    z = 4
    
    # Translate from cartesian to spherical coordinates
    r = np.sqrt( x**2 + y**2 + z**2 )
    theta = np.arccos( z/r )
    
    cosphi = x/np.sqrt( x**2 + y**2 )
    sinphi = y/np.sqrt( x**2 + y**2 )
    phi = np.arccos(cosphi)
    
    # Get magnetic fields
    X = [x,y,z]
    Bx, By, Bz = dipole_earth_cartesian(X)
    Br, Bt, Bp = dipole_earth_spherical(r, theta, phi)
    
    # Translate spherical coordinates to cartesian so we can compare
    Bxp = (Br * np.sin(theta) + Bt * np.cos(theta)) * cosphi
    Byp = (Br * np.sin(theta) + Bt * np.cos(theta)) * sinphi
    Bzp = Br * np.cos(theta) - Bt * np.sin(theta)
    
    # Compare fields
    assert( np.isclose( Bx, Bxp, 1e-5 ) )
    assert( np.isclose( By, Byp, 1e-5 ) )
    assert( np.isclose( Bz, Bzp, 1e-5 ) )
    
    logging.info('Successful dipole field test')
    
def test_dipole_function_trace():
    """Compare result from numerically tracing one field line for a simple 
    dipole model of earth's magnetic field to the analytic solution.  Dipole
    field specified by function

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    # Point where trace begins
    X0 = [ 1/2, 1/2, 1/np.sqrt(2) ]
    
    # Opposite corners of box bounding solution
    Xmin = [-100,-100,-100]
    Xmax = [100,100,100]

    # Trace field line
    field_line = trace_field_line_cartesian_function( X0, Xmin, Xmax,
                               dipole_earth_cartesian, trace_stop_earth, 
                               False, 1e-5, 0.1, 5, 'RK23')
    
    # Calculate RMS difference from analytic solution
    shape = np.shape(field_line)
    diffs = np.zeros(shape[1])
    for i in range(shape[1]):
        diffs[i] = dipole_mag_line_delta_cartesian(field_line[:,i], X0)

    rms_diffs = np.linalg.norm( diffs )/np.sqrt(len(diffs))
    
    assert( rms_diffs < 1e-4 )
    
    logging.info('Successful function dipole trace test')

def test_dipole_regular_grid_trace():
    """Compare result from numerically tracing one field line for a simple 
    dipole model of earth's magnetic field to the analytic solution.  Dipole 
    field specified through a regular grid

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    # Point where trace begins    
    X0 = [ 1/2, 1/2, 1/np.sqrt(2) ]
    
    # Get the regular grid defining the magnetic field
    [x, y, z, Bx, By, Bz] = dipole_earth_cartesian_regular_grid([5,5,5],0.1)
    
    # Trace field line
    field_line = trace_field_line_cartesian_regular_grid( X0, x, y, z, Bx, By, Bz,
                                   Stop_Function = trace_stop_earth, 
                                   forward = False, tol = 1e-5, 
                                   grid_spacing = 0.01, max_length = 5, 
                                   method_ode = 'RK23', method_interp = 'linear' )
    
    # Calculate RMS difference from analytic solution
    shape = np.shape(field_line)
    diffs = np.zeros(shape[1])
    for i in range(shape[1]):
         diffs[i] = dipole_mag_line_delta_cartesian(field_line[:,i], X0)

    rms_diffs = np.linalg.norm( diffs )/np.sqrt(len(diffs))

    assert( rms_diffs < 2e-3 )
    
    logging.info('Successful regular grid dipole trace test')

def test_dipole_unstructured_trace():
    """Compare result from numerically tracing one field line for a simple 
    dipole model of earth's magnetic field to the analytic solution.  Dipole
    field specified through an unstructured set of x,y,z points

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    
    # Point where trace begins    
    X0 = [ 1/2, 1/2, 1/np.sqrt(2) ]
    
    # Get the regular grid defining the magnetic field
    [x, y, z, Bx, By, Bz] = dipole_earth_cartesian_unstructured([5,5,5],100000)
    
    # Trace field line
    field_line = trace_field_line_cartesian_unstructured( X0, x, y, z, Bx, By, Bz,
                                   Stop_Function = trace_stop_earth, 
                                   forward = False, tol = 1e-5, 
                                   grid_spacing = 0.01, max_length = 5, 
                                   method_ode = 'RK23', method_interp = 'linear' )
    
    # Calculate RMS difference from analytic solution
    shape = np.shape(field_line)
    diffs = np.zeros(shape[1])
    for i in range(shape[1]):
        diffs[i] = dipole_mag_line_delta_cartesian(field_line[:,i], X0)

    rms_diffs = np.linalg.norm( diffs )/np.sqrt(len(diffs))

    assert( rms_diffs < 6e-2 )
    
    logging.info('Successful unstructured grid dipole trace test')
