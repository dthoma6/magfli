#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 16:42:11 2022

@author: Dean Thomas
"""

import magfli as mf
import numpy as np
import matplotlib.pyplot as plt
import logging

def demo_trace():
    """Demo function to trace one field line for a simple dipole model of earth's
    magnetic field. Solution must be outside of earth (i.e., r>1)

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
    # This box is ignored in this demo.  We use trace_stop_earth that only
    # considers whether the trace is outside of the earth.
    Xmin = [-100,-100,0]
    Xmax = [100,100,100]
    
    # Trace field line
    field_line = mf.trace_field_line_cartesian_function( X0, Xmin, Xmax,
                               Field_Function=mf.dipole_earth_cartesian, 
                               Stop_Function=mf.trace_stop_earth, 
                               forward=False, tol=1e-5, 
                               grid_spacing=0.01, max_length=5, 
                               method_ode='RK23')
    
    # Calculate RMS difference from analytic solution
    shape = np.shape(field_line)
    diffs = np.zeros(shape[1])
    for i in range(shape[1]): 
        diffs[i] = mf.dipole_mag_line_delta_cartesian(field_line[:,i], X0)

    logging.info( 'RMS difference from analytic solution: ' + 
          str(np.linalg.norm( diffs )/np.sqrt(len(diffs))) )
    logging.info( 'Max abs(diff): ' + str(max(abs(diffs))) )

    # Plot field line
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Function trace')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot( field_line[0,:], field_line[1,:], field_line[2,:] )

def demo_trace_bounded():
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
    Xmin = [-100,-100,0]
    Xmax = [100,100,100]
    
    # Trace field line
    field_line = mf.trace_field_line_cartesian_function( X0, Xmin, Xmax,
                               Field_Function=mf.dipole_earth_cartesian, 
                               Stop_Function=mf.trace_stop_box, 
                               forward=False, tol=1e-5, 
                               grid_spacing=0.01, max_length=5, 
                               method_ode='RK23')
    
    # Calculate RMS difference from analytic solution
    shape = np.shape(field_line)
    diffs = np.zeros(shape[1])
    for i in range(shape[1]): 
        diffs[i] = mf.dipole_mag_line_delta_cartesian(field_line[:,i], X0)

    logging.info( 'RMS difference from analytic solution: ' + 
          str(np.linalg.norm( diffs )/np.sqrt(len(diffs))) )
    logging.info( 'Max abs(diff): ' + str(max(abs(diffs))) )

    # Plot field line
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Bounded trace')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot( field_line[0,:], field_line[1,:], field_line[2,:] )

def demo_two_traces():
    """Demo function to trace two field lines for a simple dipole model of earth's
    magnetic field

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    # Point where trace begins
    X0 = [ 1/2, 1/2, 1/np.sqrt(2) ]
    
    # Opposite corners of box bounding domain for the solution
    # This box is ignored in this demo.  We use trace_stop_earth that only
    # considers whether the trace is outside of the earth.
    Xmin = [-100,-100,0]
    Xmax = [100,100,100]

    # Trace field line
    field_line1 = mf.trace_field_line_cartesian_function( X0, Xmin, Xmax,
                               mf.dipole_earth_cartesian, mf.trace_stop_earth, 
                               False, 1e-5, 0.01, 5, 'RK23')
    
    # Point where trace begins
    X0 = [ 1/np.sqrt(2), 0, 1/np.sqrt(2) ]
    
    # Trace field line
    field_line2 = mf.trace_field_line_cartesian_function( X0, Xmin, Xmax,
                               Field_Function=mf.dipole_earth_cartesian, 
                               Stop_Function=mf.trace_stop_earth, 
                               forward=False, tol=1e-5, 
                               grid_spacing=0.01, max_length=5, 
                               method_ode='RK23')
    
    # Plot field lines
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Two traces')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot( field_line1[0,:], field_line1[1,:], field_line1[2,:] )
    ax.plot( field_line2[0,:], field_line2[1,:], field_line2[2,:] )
    
def demo_trace_regular_grid():
    """Demo function to trace one field line for a simple dipole model of earth's
    magnetic field specified in a regular grid. Solution must be outside of 
    earth (i.e., r>1)

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
    [x, y, z, Bx, By, Bz] = mf.dipole_earth_cartesian_regular_grid([5,5,5],0.1)
    
    # Trace field line
    field_line = mf.trace_field_line_cartesian_regular_grid( X0, x, y, z, Bx, By, Bz,
                                   Stop_Function = mf.trace_stop_earth, 
                                   forward = False, tol = 1e-5, 
                                   grid_spacing = 0.1, max_length = 5, 
                                   method_ode = 'RK23', method_interp = 'nearest' )
    
    # Calculate RMS difference from analytic solution
    shape = np.shape(field_line)
    diffs = np.zeros(shape[1])
    for i in range(shape[1]):
        diffs[i] = mf.dipole_mag_line_delta_cartesian(field_line[:,i], X0)
        i += 1
    logging.info( 'RMS difference from analytic solution: ' + 
          str(np.linalg.norm( diffs )/np.sqrt(len(diffs))) )
    logging.info( 'Max abs(diff): ' + str(max(abs(diffs))) )

    # Plot field line
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Regular grid trace')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot( field_line[0,:], field_line[1,:], field_line[2,:] )

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
    [x, y, z, Bx, By, Bz] = mf.dipole_earth_cartesian_unstructured([5,5,5],100000)
    
    # Trace field line
    field_line = mf.trace_field_line_cartesian_unstructured( X0, x, y, z, Bx, By, Bz,
                                   Stop_Function = mf.trace_stop_earth, 
                                   forward = False, tol = 1e-5, 
                                   grid_spacing = 0.1, max_length = 5, 
                                   method_ode = 'RK23', method_interp = 'linear' )
    
    # Calculate RMS difference from analytic solution
    shape = np.shape(field_line)
    diffs = np.zeros(shape[1])
    for i in range(shape[1]):
        diffs[i] = mf.dipole_mag_line_delta_cartesian(field_line[:,i], X0)
 
    logging.info( 'RMS difference from analytic solution: ' + 
          str(np.linalg.norm( diffs )/np.sqrt(len(diffs))) )
    logging.info( 'Max abs(diff): ' + str(max(abs(diffs))) )

    # Plot field line
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Unstructured grid trace')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot( field_line[0,:], field_line[1,:], field_line[2,:] )


if __name__ == "__main__":
    demo_trace()
    demo_trace_bounded()
    demo_two_traces()
    demo_trace_regular_grid()
    demo_trace_unstructured()
