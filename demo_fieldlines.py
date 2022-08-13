#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:28:28 2022

@author: dean
"""

import magfli as mf
import numpy as np
import matplotlib.pyplot as plt
import time
import logging

def create_plot(title='Demo'):
    """Common procedure to generate plot in demos below
    
    Inputs
    -------
    title = title for plot
    
    Returns
    -------
    ax = reference to plot used to generate plot
    """
    
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim(-10,10)
    # ax.set_ylim(-10,10)
    # ax.set_zlim(-10,10)
    
    #ax.view_init(azim=180, elev=0)
    #ax.set_box_aspect(aspect = (2,1,1))

    return ax    
    
def get_start_points():
    """Common routine to generate start points for demos below.

    Inputs
    -------
    None.
    
    Returns
    -------
    start_pts = list of start points.
    """    
    num = 10
    start_pts = [None]*num
    x = np.linspace(-3,-10,num)
    for i in range(num) :
        start_pts[i] = [x[i],0,0]

    return start_pts    
    
def get_start_points2():
    """Common routine to generate start points for demos below.

    Inputs
    -------
    None.
    
    Returns
    -------
    start_pts = list of start points.
    """    
    num = 10
    start_pts = [None]*num
    x = np.linspace(-2,-5,num)
    for i in range(num) :
        start_pts[i] = [x[i],0,0]

    return start_pts    
    
def demo_dipole_earth_function():
    """Demo function to trace field line for a dipole magnetic field defined
    by a function.  

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """

    # Opposite corners of box bounding domain for solution
    # This box is used in this demo.  We use trace_stop_earth_box that 
    # considers whether the trace is outside the earth and inside of the box.
    Xmin = [-6,-6,-6]
    Xmax = [6,6,6]

    # Create a set of start points for fieldlines
    start_pts = get_start_points2()
            
    # Time process
    start = time.time()
    
    fl = mf.fieldlines_cartesian_function(Xmin = Xmin, Xmax = Xmax,
                    Field_Function = mf.dipole_earth_cartesian,
                    Stop_Function = mf.trace_stop_earth, 
                    tol = 1e-4, 
                    grid_spacing = 0.01, 
                    max_length = 50, 
                    method_ode = 'DOP853',
                    start_pts = None,
                    direction = None )    
    
    fl.set_start_points(start_pts, mf.integrate_direction.both)
    
    # Trace field lines
    fieldlines = fl.trace_field_lines()
    
    print('Elapsed time:' + str( time.time() - start ))
    
    # Setup plot for field lines
    ax = create_plot('Demo Function Dipole')    

    # Plot fieldlines    
    num = len(fieldlines)
    for i in range(num):
        # Plot field line
        ax.plot( fieldlines[i][0,:], fieldlines[i][1,:], fieldlines[i][2,:], color='blue' )

def demo_dipole_earth_regular_grid():
    """Demo function to trace field line for a dipole magnetic field defined
    by a regular grid.  

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """

    # Get the regular grid defining the magnetic field
    logging.info('Initializing regular grid' )
    x, y, z, Bx, By, Bz = mf.dipole_earth_cartesian_regular_grid([6,6,6],0.3)
    logging.info('Done with regular grid' )

    # Create a set of start points for fieldlines
    start_pts = get_start_points2()
            
    # Time process
    start = time.time()
    
    fl = mf.fieldlines_cartesian_regular_grid(x = x, y = y, z = z,
                    Fx = Bx, Fy = By, Fz = Bz,
                    Stop_Function = mf.trace_stop_earth_box, 
                    tol = 1e-4, 
                    grid_spacing = 0.01, 
                    max_length = 50, 
                    method_ode = 'RK45',
                    method_interp = 'linear',
                    start_pts = start_pts,
                    direction = mf.integrate_direction.both )    
    
    # Trace field lines
    fieldlines = fl.trace_field_lines()
    
    print('Elapsed time:' + str( time.time() - start ))
    
    # Setup plot for field lines
    ax = create_plot('Demo Regular Grid Dipole')    

    # Plot fieldlines    
    num = len(fieldlines)
    for i in range(num):
        # Plot field line
        ax.plot( fieldlines[i][0,:], fieldlines[i][1,:], fieldlines[i][2,:], color='blue' )

def demo_dipole_earth_unstructured():
    """Demo function to trace field line for a dipole magnetic field defined by
    an unstructured grid.  

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """

    # Opposite corners of box bounding domain for solution
    # This box is used in this demo.  We use trace_stop_earth_box that 
    # considers whether the trace is outside the earth and inside of the box.
    Xmin = [-6,-6,-6]
    Xmax = [6,6,6]

    # Get the unstructured grid defining the magnetic field, grid contains
    # num_pts randomly located points
    logging.info('Initializing unstructured grid' )
    x, y, z, Bx, By, Bz = mf.dipole_earth_cartesian_unstructured(Xmax,10000)
    logging.info('Done with unstructured grid' )

    # Create a set of start points for fieldlines
    start_pts = get_start_points2()
            
    # Time process
    start = time.time()
    
    fl = mf.fieldlines_cartesian_unstructured(x = x, y = y, z = z,
                    Fx = Bx, Fy = By, Fz = Bz,
                    Stop_Function = mf.trace_stop_earth, 
                    tol = 1e-4, 
                    grid_spacing = 0.01, 
                    max_length = 50, 
                    method_ode = 'DOP853',
                    method_interp = 'linear',
                    start_pts = None,
                    direction = None )    
    
    fl.set_start_points(start_pts, mf.integrate_direction.both)
    
    # Trace field lines
    fieldlines = fl.trace_field_lines()
    
    print('Elapsed time:' + str( time.time() - start ))
    
    # Setup plot for field lines
    ax = create_plot('Demo Unstructured Dipole')    

    # Plot fieldlines    
    num = len(fieldlines)
    for i in range(num):
        # Plot field line
        ax.plot( fieldlines[i][0,:], fieldlines[i][1,:], fieldlines[i][2,:], color='blue' )

def demo_BATSRUS():
    """Demo function to trace field line for a magnetic field stored in
    a BATSRUS file.  

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """

    # File used in demo
    filename = '/tmp/3d__var_2_e20190902-041000-000'
 
    # Create a set of start points for fieldlines
    start_pts = get_start_points()
            
    # Time process
    start = time.time()
    
    fl = mf.fieldlines_cartesian_unstructured_BATSRUSfile(filename = filename,
                    Stop_Function = mf.trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest',
                    start_pts = None,
                    direction = None )
    
    fl.set_start_points(start_pts, mf.integrate_direction.both)
    
    # Setup multitrace for tracing field lines
    fieldlines = fl.trace_field_lines()
    
    print('Elapsed time:' + str( time.time() - start ))
    
    # Setup plot for field lines
    ax = create_plot('Demo BATSRUS file')    

    # Plot fieldlines    
    num = len(fieldlines)
    for i in range(num):
        # Plot field line
        ax.plot( fieldlines[i][0,:], fieldlines[i][1,:], fieldlines[i][2,:], color='blue' )

def demo_swmfio_BATSRUS():
    """Demo function to trace field line for a magnetic field stored in
    a BATSRUS file.  

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """

    # File used in demo
    filename = '/tmp/3d__var_2_e20190902-041000-000'

    # Create a set of start points for fieldlines
    start_pts = get_start_points()
            
    # Time process
    start = time.time()
    
    fl = mf.fieldlines_cartesian_unstructured_swmfio_BATSRUSfile(filename = filename,
                    Stop_Function = mf.trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'swmfio',
                    start_pts = None,
                    direction = None )
    
    fl.set_start_points(start_pts, mf.integrate_direction.both)
    
    # Setup multitrace for tracing field lines
    fieldlines = fl.trace_field_lines()
    
    print('Elapsed time:' + str( time.time() - start ))
    
    # Setup plot for field lines
    ax = create_plot('Demo BATSRUS file, swmfio interpolator')    

    # Plot fieldlines    
    num = len(fieldlines)
    for i in range(num):
        # Plot field line
        ax.plot( fieldlines[i][0,:], fieldlines[i][1,:], fieldlines[i][2,:], color='blue' )

def demo_VTK():
    """Demo function to trace field line for a magnetic field stored in
    a VTK file.  

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    # File used in demo
    filename = "/tmp/3d__var_2_e20190902-041000-000.vtk"
    
    # Create a set of start points for fieldlines
    start_pts = get_start_points()
                
    # Time process
    start = time.time()
    
    fl = mf.fieldlines_cartesian_unstructured_VTKfile(filename = filename,
                    Stop_Function = mf.trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest',
                    start_pts = None,
                    direction = None,
                    F_field = 'b',
                    cell_centers = None)
    
    fl.set_start_points(start_pts, mf.integrate_direction.both)
    
    # Setup multitrace for tracing field lines
    fieldlines = fl.trace_field_lines()
    
    print('Elapsed time:' + str( time.time() - start ))
    
    # Setup plot for field lines
    ax = create_plot('Demo VTK file')    

    # Plot fieldlines    
    num = len(fieldlines)
    for i in range(num):
        # Plot field line
        ax.plot( fieldlines[i][0,:], fieldlines[i][1,:], fieldlines[i][2,:], color='blue' )
        
def demo_paraview_VTK():
    """Demo function to trace field line for a magnetic field stored in
    a VTK file.  

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    # File used in demo
    filename = "/tmp/3d__var_2_e20190902-041000-000.vtk"
    
    # Create a set of start points for fieldlines
    #seed_type = 'Line' 
    seed_type = 'Line' 
    start_pt1 = [-3,0,0]
    # seed_type = 'Point Cloud' 
    # start_pt1 = [0,0,0]
    start_pt2 = [-10,0,0]
    radius = 1
    resolution = 9
 
    # Time process
    start = time.time()
    
    fl = mf.fieldlines_cartesian_unstructured_paraview_VTKfile(filename = filename,
                    max_length = 100, 
                    method_ode = 'RK2',
                    seed_type = seed_type,
                    start_pt1 = start_pt1,
                    start_pt2 = start_pt2,
                    radius = radius,
                    resolution = resolution,
                    direction = mf.integrate_direction.both,
                    F_field = 'b')
    
    # # Setup multitrace for tracing field lines
    fieldlines = fl.trace_field_lines()
     
    print('Elapsed time:' + str( time.time() - start ))
    
    # Get rid of paraview items
    fl.reset()
    
    # Setup plot for field lines
    ax = create_plot('Demo Paraview VTK file')    

    # Plot fieldlines    
    num = len(fieldlines)
    for i in range(num):
        # Plot field line
        ax.plot( fieldlines[i][0,:], fieldlines[i][1,:], fieldlines[i][2,:], color='blue' )


if __name__ == "__main__":
    # demo_dipole_earth_function()
    # demo_dipole_earth_regular_grid()
    # demo_dipole_earth_unstructured()
    # demo_BATSRUS()
    # demo_swmfio_BATSRUS()
    # demo_VTK()
    demo_paraview_VTK()
    
