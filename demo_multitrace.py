#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 08:03:11 2022

@author: Dean Thomas
"""
import magfli as mf
import numpy as np
import matplotlib.pyplot as plt
import logging
import time

def demo_trace_function():
    """Demo function to trace one field line for a simple dipole model of earth's
    magnetic field. Solution must be inside box with corners Xmin and Xmax

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    # Point where trace begins
    X0 = [ 1/2, 1/2, 1/np.sqrt(2) ]
    
    # Opposite corners of box bounding domain for solution
    # This box is used in this demo.  We use trace_stop_box that only
    # considers whether the trace is inside of the box, and ignores the earth.
    Xmin = [-100,-100,100]
    Xmax = [100,100,100]
    
    # Setup multitrace
    mt = mf.multitrace_cartesian_function( Xmin, Xmax,
                                   Field_Function = mf.dipole_earth_cartesian,
                                   Stop_Function = mf.trace_stop_earth, 
                                   tol = 1e-5, grid_spacing = 0.1, max_length = 5, 
                                   method_ode = 'RK23' )
    
    # Trace field line
    field_line = mt.trace_field_line( X0, False )
    
    # Calculate RMS difference from analytic solution
    shape = np.shape(field_line)
    diffs = np.zeros(shape[1])
    for i in range(shape[1]): 
        diffs[i] = mf.dipole_mag_line_delta_cartesian(field_line[:,i], X0)

    logging.info( 'RMS difference from analytic solution: ' + 
          str(np.linalg.norm( diffs )/np.sqrt(len(diffs))) )
    logging.info( 'Max abs(diff): ' + str(max(abs(diffs))) )

    # Plot field line
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title('Function trace')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot( field_line[0,:], field_line[1,:], field_line[2,:] )
    
    return

def demo_trace_regular_grid():
    """Demo function to trace one field line for a simple dipole model of earth's
    magnetic field specified in a regular grid of points. Solution must be 
    outside of earth (i.e., r>1)

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
    X, Y, Z, Bx, By, Bz = mf.dipole_earth_cartesian_regular_grid([5,5,5],0.1)
        
    # Setup multitrace
    mt = mf.multitrace_cartesian_regular_grid( X, Y, Z, Bx, By, Bz,
                                   Stop_Function = mf.trace_stop_earth, 
                                   tol = 1e-5, grid_spacing = 0.1, max_length = 5, 
                                   method_ode = 'RK23', method_interp = 'linear' )
    
    # Trace field line
    field_line = mt.trace_field_line( X0, False )
    
    # Calculate RMS difference from analytic solution
    shape = np.shape(field_line)
    diffs = np.zeros(shape[1])
    for i in range(shape[1]):
        diffs[i] = mf.dipole_mag_line_delta_cartesian(field_line[:,i], X0)
 
    logging.info( 'RMS difference from analytic solution: ' + 
          str(np.linalg.norm( diffs )/np.sqrt(len(diffs))) )
    logging.info( 'Max abs(diff): ' + str(max(abs(diffs))) )

    # Plot field line
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title('Regular grid trace')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot( field_line[0,:], field_line[1,:], field_line[2,:] )
    
    return

def demo_trace_unstructured():
    """Demo function to trace one field line for a simple dipole model of earth's
    magnetic field specified in an unstructured set of points. Solution must 
    be outside of earth (i.e., r>1)

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
    X, Y, Z, Bx, By, Bz = mf.dipole_earth_cartesian_unstructured([5,5,5],100000)
    
    # Setup multitrace
    mt = mf.multitrace_cartesian_unstructured( X, Y, Z, Bx, By, Bz,
                                   Stop_Function = mf.trace_stop_earth, 
                                   tol = 1e-5, grid_spacing = 0.1, max_length = 5, 
                                   method_ode = 'RK23', method_interp = 'linear' )
    
    # Trace field line
    field_line = mt.trace_field_line( X0, False )
    
    # Calculate RMS difference from analytic solution
    shape = np.shape(field_line)
    diffs = np.zeros(shape[1])
    for i in range(shape[1]):
        diffs[i] = mf.dipole_mag_line_delta_cartesian(field_line[:,i], X0)
 
    logging.info( 'RMS difference from analytic solution: ' + 
          str(np.linalg.norm( diffs )/np.sqrt(len(diffs))) )
    logging.info( 'Max abs(diff): ' + str(max(abs(diffs))) )

    # Plot field line
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title('Unstructured grid trace')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot( field_line[0,:], field_line[1,:], field_line[2,:] )
    
    return

if __name__ == "__main__":
    demo_trace_function()
    demo_trace_regular_grid()
    demo_trace_unstructured()
