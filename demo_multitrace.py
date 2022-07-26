#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 16:42:11 2022

@author: Dean Thomas
"""

import magfli as mf
import numpy as np
import matplotlib.pyplot as plt
import time

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
    # Opposite corners of box bounding domain for the solution
    # This box is ignored in this demo.  We use trace_stop_earth that only
    # considers whether the trace is outside of the earth.
    Xmin = [-100,-100,-100]
    Xmax = [100,100,100]

    # Setup multitrace
    mt = mf.multitrace_cartesian_function( Xmin, Xmax,
                                   Field_Function = mf.dipole_earth_cartesian,
                                   Stop_Function = mf.trace_stop_earth, 
                                   tol = 1e-5, grid_spacing = 0.1, max_length = 5, 
                                   method_ode = 'RK23' )
    
    # Point where first trace begins
    X0 = [ 1/2, 1/2, 1/np.sqrt(2) ]
    
    # Trace first field line.  Note, forward = False so field line is r > 1.
    field_line1 = mt.trace_field_line( X0, False )
    
    # Second point where trace begins
    X0 = [ 1/np.sqrt(2), 0, 1/np.sqrt(2) ]
    
    # Trace second field line.  Again, forward = False.
    field_line2 = mt.trace_field_line( X0, False )
        
    # Plot field lines
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title('Two traces')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot( field_line1[0,:], field_line1[1,:], field_line1[2,:] )
    ax.plot( field_line2[0,:], field_line2[1,:], field_line2[2,:] )
    
    return
    

def demo_trace_dipole_earth_function():
    """Demo function to trace multiple field lines for a simple dipole model 
    of earth's magnetic field specified by a function. Solution must be outside
    of earth (i.e., r>1) and within the box bounding the domain.  Start points 
    for the field lines will be distributed over the earth and the bounding box.

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
    Xmin = [-6,-3,-3]
    Xmax = [6,3,3]
    
    # Time process
    start = time.time()

    # Setup plot for field lines
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title('Multiple Field Lines, Dipole Function')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(Xmin[0],Xmax[0])
    ax.set_ylim(Xmin[1],Xmax[1])
    ax.set_zlim(Xmin[2],Xmax[2])
    #ax.view_init(azim=0, elev=90)
    ax.set_box_aspect(aspect = (2,1,1))
    
    # Setup multitrace
    mt = mf.multitrace_cartesian_function( Xmin, Xmax,
                                   Field_Function = mf.dipole_earth_cartesian,
                                   Stop_Function = mf.trace_stop_earth_box, 
                                   tol = 1e-4, grid_spacing = 0.1, max_length = 5, 
                                   method_ode = 'DOP853' )

    # Internal function used for repetitive work
    def trace_and_plot(forward):
        # Trace field line
        field_line = mt.trace_field_line( X0, forward )

        # Plot field line
        ax.plot( field_line[0,:], field_line[1,:], field_line[2,:], color='blue' )
        
        return field_line

    # We'll walk around northern hemisphere in 20 degree increments
    # Note: forward has to be False in northern hemisphere for the call to 
    # trace_field_line...

    # Colatitude array in radians
    colatitude = np.deg2rad([20,30,40,60,80])
    # Longitude array in radians
    longitude = np.deg2rad([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 
                            220, 240, 260, 280, 300, 320, 340])
    
    # Create storage for field lines that do NOT return to earth.  That is, they
    # terminate because they exceed the bounds specified by Xmin and Xmax.  We
    # will want to start these from the southern hemisphere to "complete" the
    # field line
    southern = np.empty((0,3), float)
    
    # Walk around northern hemisphere
    for col in colatitude:
        for long in longitude:
            # Point where trace begins
            X0 = [ np.cos(long)*np.sin(col), np.sin(long)*np.sin(col), np.cos(col) ]
            # Trace field line
            field_line = trace_and_plot(False)
            # Identified terminated field lines. i.e., r > 1 (outside earth)
            if( np.linalg.norm( field_line[:,-1] ) > 1. ):
                southern = np.append(southern, np.array([[X0[0], X0[1], -X0[2]]]), axis=0)
            
    # We'll walk around southern hemisphere filling in the field lines from
    # the northern hemisphere that terminated before returning to earth 
    # Note: forward has to be True in southern hemisphere
    
    for X0 in southern:
        trace_and_plot(True)

    
    # Walk across surface of bounding box defined by Xmin and Xmax.  Trace field
    # lines coming in through the surface.  Start by defining x,y,z arrays 
    # defining points those surfaces, ignoring the edges.  Loop through the arrays
    
    x = np.linspace(Xmin[0]+1,Xmax[0]-1,10)
    y = np.linspace(Xmin[1]+1,Xmax[1]-1,5)
    z = np.linspace(Xmin[1]+1,Xmax[2]-1,5)

    # Do full x-y plane top and bottom
    # Note: we determine which way to go along the line forward (+step) or
    # not forward (-step) for each point.  We want the field line to grow
    # into the bounding box
    for xx in x:
        for yy in y:
            # Top surface
            X0 = [xx, yy, Xmax[2]]
            # We want Bz < 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[2] < 0
            # Trace field line
            trace_and_plot(forward)
            # Bottom surface
            X0 = [xx, yy, Xmin[2]]
            # We want Bz > 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[2] > 0
            # Trace field line
            trace_and_plot(forward)

    # Do both x-z planes.
    for xx in x:
        for zz in z:
            # Top surface
            X0 = [xx, Xmax[1], zz]
            # We want By < 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[1] < 0
            # Trace field line
            trace_and_plot(forward)
            # Bottom surface
            X0 = [xx, Xmin[1], zz]
            # We want By > 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[1] > 0
            # Trace field line
            trace_and_plot(forward)

    # Do both y-z planes.
    for yy in y:
        for zz in z:
            # Top surface
            X0 = [Xmax[0], yy, zz]
            # We want Bx < 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[0] < 0
            # Trace field line
            trace_and_plot(forward)
            # Bottom surface
            X0 = [Xmin[0], yy, zz]
            # We want Bz > 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[0] > 0
            # Trace field line
            trace_and_plot(forward)
        
    print('Elapsed time:' + str( time.time() - start ))
    
    return
            
def demo_trace_dipole_earth_unstructured():
    """Demo function to trace multiple field lines for a simple dipole model 
    of earth's magnetic field specified by an unstructured grid. Solution must 
    be outside of earth (i.e., r>1) and within the box bounding the domain.  
    Start points for the field line will be distributed over the earth and the 
    bounding box.

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
    Xmin = [-6,-3,-3]
    Xmax = [6,3,3]
    
    # Get the unstructured grid defining the magnetic field, grid contains
    # num_pts randomly located points
    num_pts = 100000
    X, Y, Z, Bx, By, Bz = mf.dipole_earth_cartesian_unstructured(Xmax,num_pts)
    
    # Time process
    start = time.time()

    # Setup plot for field lines
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title('Multiple Field Lines, Dipole Unstructured, ' + str(num_pts) + ' pts')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(Xmin[0],Xmax[0])
    ax.set_ylim(Xmin[1],Xmax[1])
    ax.set_zlim(Xmin[2],Xmax[2])
    #ax.view_init(azim=0, elev=90)
    ax.set_box_aspect(aspect = (2,1,1))

    # Setup multitrace for tracing field lines
    mt = mf.multitrace_cartesian_unstructured( X, Y, Z, Bx, By, Bz,
                                   Stop_Function = mf.trace_stop_earth_box, 
                                   tol = 1e-4, grid_spacing = 0.1, max_length = 5, 
                                   method_ode = 'DOP853', method_interp = 'nearest' )
    
   # Internal function used for the repetitive work of tracing field lines
    def trace_and_plot(forward):
        # Trace field line
        field_line = mt.trace_field_line( X0, forward )

        # Plot field line
        ax.plot( field_line[0,:], field_line[1,:], field_line[2,:], color='blue' )
        
        return field_line

    # We'll walk around northern hemisphere in 20 degree increments
    # Note: forward has to be False in northern hemisphere for the call to 
    # trace_field_line... so that the field line trace goes away from the earth
    
    # Colatitude array in radians
    colatitude = np.deg2rad([20,30,40,60,80])
    # Longitude array in radians
    longitude = np.deg2rad([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 
                            220, 240, 260, 280, 300, 320, 340])
    
    # Create storage for field lines that do NOT return to earth.  That is, they
    # terminate because they exceed the bounds specified by Xmin and Xmax.  We
    # will want to start these from the southern hemisphere to "complete" the
    # field line
    southern = np.empty((0,3), float)
    
    # Walk around northern hemisphere
    for col in colatitude:
        for long in longitude:
            # Point where trace begins
            X0 = [ np.cos(long)*np.sin(col), np.sin(long)*np.sin(col), np.cos(col) ]
            # Trace field line.  As noted above, forward = False
            field_line = trace_and_plot(False)
            # Identified terminated field lines, i.e. r > 1 (outside earth)
            if( np.linalg.norm( field_line[:,-1] ) > 1. ):
                southern = np.append(southern, np.array([[X0[0], X0[1], -X0[2]]]), axis=0)
            
    # We'll walk around southern hemisphere filling in the field lines from
    # the northern hemisphere that terminated before returning to earth 
    # Note: forward has to be True in southern hemisphere.  Going forward
    # heads away from the earth
    for X0 in southern:
        # Trace field line
        trace_and_plot(True)

    
    # Walk across surface of bounding box defined by Xmin and Xmax.  Trace field
    # lines coming in through the surface.  Start by defining x,y,z arrays 
    # defining points those surfaces, ignoring the edges.  Loop through the arrays
    x = np.linspace(Xmin[0]+1,Xmax[0]-1,10)
    y = np.linspace(Xmin[1]+1,Xmax[1]-1,5)
    z = np.linspace(Xmin[1]+1,Xmax[2]-1,5)

    # Do both x-y planes, top and bottom.
    # Note: we determine which way to go along the line forward (+step) or
    # not forward (-step) for each point.  We want the field line to grow
    # into the bounding box
    for xx in x:
        for yy in y:
            # Top surface
            X0 = [xx, yy, Xmax[2]]
            # We want Bz < 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[2] < 0
            # Trace field line
            trace_and_plot(forward)
            # Bottom surface
            X0 = [xx, yy, Xmin[2]]
            # We want Bz > 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[2] > 0
            # Trace field line
            trace_and_plot(forward)

    # Do both x-z planes.
    for xx in x:
        for zz in z:
            # Top surface
            X0 = [xx, Xmax[1], zz]
            # We want By < 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[1] < 0
            # Trace field line
            trace_and_plot(forward)
            # Bottom surface
            X0 = [xx, Xmin[1], zz]
            # We want By > 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[1] > 0
            # Trace field line
            trace_and_plot(forward)

    # Do both y-z planes.
    for yy in y:
        for zz in z:
            # Top surface
            X0 = [Xmax[0], yy, zz]
            # We want Bx < 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[0] < 0
            # Trace field line
            trace_and_plot(forward)
            # Bottom surface
            X0 = [Xmin[0], yy, zz]
            # We want Bz > 0 for forward tracing (forward = True)
            forward = (mf.dipole_earth_cartesian(X0))[0] > 0
            # Trace field line
            trace_and_plot(forward)
        
    print('Elapsed time:' + str( time.time() - start ))
    
    return
            
def demo_trace_dipole_earth_unstructured_file():
    """Demo function to trace multiple field lines for a model of the  
    earth's magnetic field specified by an unstructured grid from a
    SWMF output file.  Solution must be outside of earth (i.e., r>1) and within 
    the box bounding the domain. Start points for the field line will be 
    distributed over the earth and the bounding box.

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    from os.path import exists
    import swmfio as swmfio

    # Location of SWMF file
    tmpdir = '/Volumes/Physics HD/runs/DIPTSUR2/GM/IO2/'
    filebase = '3d__var_2_e20190902-041400-000'
        
    print("Reading " + tmpdir + filebase + ".{tree, info, out}")
    batsclass = swmfio.read_batsrus(tmpdir + filebase)
    
    # Get x, y, z and Bx, By, Bz from file
    var_dict = dict(batsclass.varidx)
    X = batsclass.data_arr[:,var_dict['x']][:]
    Y = batsclass.data_arr[:,var_dict['y']][:]
    Z = batsclass.data_arr[:,var_dict['z']][:]
     
    Bx = batsclass.data_arr[:,var_dict['bx']][:]
    By = batsclass.data_arr[:,var_dict['by']][:]
    Bz = batsclass.data_arr[:,var_dict['bz']][:]

    # Opposite corners of box bounding domain for solution
    # This box is used in this demo.  We use trace_stop_earth_box that 
    # considers whether the trace is outside the earth and inside of the box.
    Xmin = [min(X), min(Y), min(Z)]
    Xmax = [max(X), max(Y), max(Z)]
    
    # Count the number of field lines 
    cnt = 0
    
    # Time process
    start = time.time()

    # Setup plot for field lines
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title('Multiple Field Lines, Unstructured SWMF')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(Xmin[0],Xmax[0])
    ax.set_ylim(Xmin[1],Xmax[1])
    ax.set_zlim(Xmin[2],Xmax[2])
    #ax.view_init(azim=180, elev=0)
    ax.set_box_aspect(aspect = (2,1,1))

    # Setup multitrace for tracing field lines
    mt = mf.multitrace_cartesian_unstructured( X, Y, Z, Bx, By, Bz,
                                   Stop_Function = mf.trace_stop_earth_box, 
                                   tol = 1e-4, grid_spacing = 0.1, max_length = 500, 
                                   method_ode = 'DOP853', method_interp = 'nearest' )

    # Internal function used for the repetitive work of tracing field lines
    def trace_and_plot(forward):
        # Trace field line
        field_line = mt.trace_field_line( X0, forward )
                
        # Plot field line
        ax.plot( field_line[0,:], field_line[1,:], field_line[2,:], color='blue' )
 
        return field_line

    # We'll walk around northern hemisphere in 20 degree increments
    # Note: forward has to be False in northern hemisphere for the call to 
    # trace_field_line... so that the field line trace goes away from the earth
    
    # Colatitude array in radians
    colatitude = np.deg2rad([20,30,40,60,80])
    
    # Longitude array in radians
    longitude = np.deg2rad([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 
                            220, 240, 260, 280, 300, 320, 340])
    
    # Create storage for field lines that do NOT return to earth.  That is, they
    # terminate because they exceed the bounds specified by Xmin and Xmax.  We
    # will want to start these from the southern hemisphere to "complete" the
    # field line
    southern = np.empty((0,3), float)
  
    # Walk around northern hemisphere
    for col in colatitude:
        for long in longitude:
            # Point where trace begins
            X0 = [ np.cos(long)*np.sin(col), np.sin(long)*np.sin(col), np.cos(col) ]
            # Trace field line
            field_line = trace_and_plot(False)
            # Identified terminated field lines, i.e. r > 1 (outside earth)
            if( np.linalg.norm( field_line[:,-1] ) > 1. ):
                southern = np.append(southern, np.array([[X0[0], X0[1], -X0[2]]]), axis=0)
            cnt += 1
            
    # We'll walk around southern hemisphere filling in the field lines from
    # the northern hemisphere that terminated before returning to earth 
    # Note: forward has to be True in southern hemisphere to head away from the
    # earth
    for X0 in southern:
        # Trace field line
        trace_and_plot(True)
        cnt += 1

    # Walk across surface of bounding box defined by Xmin and Xmax.  Trace field
    # lines coming in through the surface.  Start by defining x,y,z arrays 
    # defining points on the surfaces, ignoring the edges.  Loop through the arrays
    x = np.linspace(int(Xmin[0]+1),int(Xmax[0]-1),10)
    y = np.linspace(int(Xmin[1]+1),int(Xmax[1]-1),5)
    z = np.linspace(int(Xmin[2]+1),int(Xmax[2]-1),5)

    # Do both x-y planes, top and bottom.
    # Note: we determine which way to go along the line forward (+step) or
    # not forward (-step) for each point.  We want the field line to grow
    # into the bounding box
    for xx in x:
        for yy in y:
            # Top surface
            X0 = [xx, yy, Xmax[2]]
            # We want Bz < 0 for forward tracing (forward = True)
            forward = (mt.trace_field_value(X0))[2] < 0
            # Trace field line
            trace_and_plot(forward)
            # Bottom surface
            X0 = [xx, yy, Xmin[2]]
            # We want Bz > 0 for forward tracing (forward = True)
            forward = (mt.trace_field_value(X0))[2] > 0
            # Trace field line
            trace_and_plot(forward)
            cnt += 2

    # Do both x-z planes.
    for xx in x:
        for zz in z:
            # Top surface
            X0 = [xx, Xmax[1], zz]
            # We want By < 0 for forward tracing (forward = True)
            forward = (mt.trace_field_value(X0))[1] < 0
            # Trace field line
            trace_and_plot(forward)
            # Bottom surface
            X0 = [xx, Xmin[1], zz]
            # We want By > 0 for forward tracing (forward = True)
            forward = (mt.trace_field_value(X0))[1] > 0
            # Trace field line
            trace_and_plot(forward)
            cnt += 2

    # Do both y-z planes.
    for yy in y:
        for zz in z:
            # Top surface
            X0 = [Xmax[0], yy, zz]
            # We want Bx < 0 for forward tracing (forward = True)
            forward = (mt.trace_field_value(X0))[0] < 0
            # Trace field line
            trace_and_plot(forward)
            # Bottom surface
            X0 = [Xmin[0], yy, zz]
            # We want Bz > 0 for forward tracing (forward = True)
            forward = (mt.trace_field_value(X0))[0] > 0
            # Trace field line
            trace_and_plot(forward)
            cnt += 2
        
    print('Elapsed time:' + str( time.time() - start ))
    print('Total field lines: ', str(cnt))
    
    return

def demo_trace_dipole_earth_unstructured_file_line():
    """Demo function to trace multiple field lines for a model of the  
    earth's magnetic field specified by an unstructured grid from a
    SWMF output file.  Solution must be outside of earth (i.e., r>1) and within 
    the box bounding the domain. Start points for the field lines will be 
    over a series of points along a line.

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    from os.path import exists
    import swmfio as swmfio

    # Location of SWMF file
    tmpdir = '/Volumes/Physics HD/runs/DIPTSUR2/GM/IO2/'
    filebase = '3d__var_2_e20190902-041400-000'
        
    print("Reading " + tmpdir + filebase + ".{tree, info, out}")
    batsclass = swmfio.read_batsrus(tmpdir + filebase)
    
    # Get x, y, z and Bx, By, Bz from file
    var_dict = dict(batsclass.varidx)
    X = batsclass.data_arr[:,var_dict['x']][:]
    Y = batsclass.data_arr[:,var_dict['y']][:]
    Z = batsclass.data_arr[:,var_dict['z']][:]
     
    Bx = batsclass.data_arr[:,var_dict['bx']][:]
    By = batsclass.data_arr[:,var_dict['by']][:]
    Bz = batsclass.data_arr[:,var_dict['bz']][:]

    # Opposite corners of box bounding domain for solution
    # This box is used in this demo.  We use trace_stop_earth_box that 
    # considers whether the trace is outside the earth and inside of the box.
    Xmin = [min(X), min(Y), min(Z)]
    Xmax = [max(X), max(Y), max(Z)]
    
    # Count the number of field lines 
    cnt = 0
    
    # Time process
    start = time.time()

    # Setup plot for field lines
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title(filebase)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_xlim(Xmin[0],Xmax[0])
    #ax.set_ylim(Xmin[1],Xmax[1])
    #ax.set_zlim(Xmin[2],Xmax[2])
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    #ax.view_init(azim=180, elev=0)
    #ax.set_box_aspect(aspect = (2,1,1))

    # Setup multitrace for tracing field lines
    mt = mf.multitrace_cartesian_unstructured( X, Y, Z, Bx, By, Bz,
                                   Stop_Function = mf.trace_stop_earth_box, 
                                   tol = 1e-4, grid_spacing = 0.1, max_length = 500, 
                                   method_ode = 'RK45', method_interp = 'nearest' )

    # Internal function used for the repetitive work of tracing field lines
    def trace_and_plot(forward):
        # Trace field line
        field_line = mt.trace_field_line( X0, forward )
                
        # Plot field line
        ax.plot( field_line[0,:], field_line[1,:], field_line[2,:], color='blue' )
 
        return field_line

    # Create the start points for the field lines.  10 points along a line
    # from [-3, 0, 0] to [-10, 0, 0]
    # We'll walk along line, and do integration in both the forward and backwards
    # directions
  
    for x in range(10):
        # Point where trace begins
        X0 = [-3-x*7/9, 0, 0]
        # Trace field line
        field_line = trace_and_plot(True)
        field_line = trace_and_plot(False)
        cnt += 2
            
    print('Elapsed time:' + str( time.time() - start ))
    print('Total field lines: ', str(cnt))
    
    return

def demo_spacing_on_box_surfaces():
    """Demo function to look at options for exterior points on bounding box.

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
    Xmin = [-6,-3,-3]
    Xmax = [6,3,3]
    
    # Setup plot for field lines
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title('Test exterior points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(Xmin[0],Xmax[0])
    ax.set_ylim(Xmin[1],Xmax[1])
    ax.set_zlim(Xmin[2],Xmax[2])
    #ax.view_init(azim=0, elev=90)
    ax.set_box_aspect(aspect = (2,1,1))
    
    
    # Walk across surface of bounding box defined by Xmin and Xmax.  Trace field
    # lines coming in through the surface.  Start by defining x,y,z arrays 
    # defining points those surfaces, ignoring the edges.  Loop through the arrays
    
    x = np.linspace(Xmin[0]+1,Xmax[0]-1,10)
    y = np.linspace(Xmin[1]+1,Xmax[1]-1,5)
    z = np.linspace(Xmin[1]+1,Xmax[2]-1,5)
    
    # Do both x-y planes, top and bottom
    for xx in x:
        for yy in y:
             # Top surface
             X0 = [xx, yy, Xmax[2]]
             ax.scatter(X0[0], X0[1], X0[2], c='r')
             # Bottom surface
             X0 = [xx, yy, Xmin[2]]
             ax.scatter(X0[0], X0[1], X0[2], c='r')  
    
    # Do both x-z planes.
    for xx in x:
        for zz in z:
             # Top surface
             X0 = [xx, Xmax[1], zz]
             ax.scatter(X0[0], X0[1], X0[2], c='r')
             # Bottom surface
             X0 = [xx, Xmin[1], zz]
             ax.scatter(X0[0], X0[1], X0[2], c='r')
    
    # Do both y-z planes. 
    for yy in y:
        for zz in z:
             # Top surface
             X0 = [Xmax[0], yy, zz]
             ax.scatter(X0[0], X0[1], X0[2], c='r')
             # Bottom surface
             X0 = [Xmin[0], yy, zz]
             ax.scatter(X0[0], X0[1], X0[2], c='r')

    return

if __name__ == "__main__":
    # demo_two_traces()
    # demo_trace_dipole_earth_function()
    # demo_trace_dipole_earth_unstructured()
    # demo_trace_dipole_earth_unstructured_file()
    demo_trace_dipole_earth_unstructured_file_line()
    # demo_spacing_on_box_surfaces()
