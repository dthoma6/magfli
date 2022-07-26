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

def create_plot(title='Demo'):
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    
    #ax.view_init(azim=180, elev=0)
    #ax.set_box_aspect(aspect = (2,1,1))

    return ax    
    
    
def demo_trace_BATSRUS():
    """Demo function to trace field line for a magnetic field stored in
    a BATSRUS file.  

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """

    filename = '/tmp/3d__var_2_e20190902-041400-000'

    num = 10
    startPts = [None]*num
    x = np.linspace(-3,-10,num)
    for i in range(num) :
        startPts[i] = [x[i],0,0]
            
    # Setup plot for field lines
    ax = create_plot('Demo BATSRUS file')    
    
    # Time process
    start = time.time()
    
    fl = mf.fieldlines_cartesian_unstructured_BATSRUSfile(filename = filename,
                    Stop_Function = mf.trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest',
                    startPts = None,
                    direction = None )
    
    fl.setstartpoints(startPts, mf.integrate_direction.both)
    
    # Setup multitrace for tracing field lines
    fieldlines = fl.tracefieldlines()
    
    print('Elapsed time:' + str( time.time() - start ))
    
    num = len(fieldlines)
    for i in range(num):
        # Plot field line
        ax.plot( fieldlines[i][0,:], fieldlines[i][1,:], fieldlines[i][2,:], color='blue' )

def demo_trace_swmfio_BATSRUS():
    """Demo function to trace field line for a magnetic field stored in
    a BATSRUS file.  

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """

    filename = '/tmp/3d__var_2_e20190902-041400-000'

    num = 10
    startPts = [None]*num
    x = np.linspace(-3,-10,num)
    for i in range(num) :
        startPts[i] = [x[i],0,0]
            
    # Setup plot for field lines
    ax = create_plot('Demo BATSRUS file, SWMFIO interpolate')    
    
    # Time process
    start = time.time()
    
    fl = mf.fieldlines_cartesian_unstructured_swmfio_BATSRUSfile(filename = filename,
                    Stop_Function = mf.trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest',
                    startPts = None,
                    direction = None )
    
    fl.setstartpoints(startPts, mf.integrate_direction.both)
    
    # Setup multitrace for tracing field lines
    fieldlines = fl.tracefieldlines()
    
    print('Elapsed time:' + str( time.time() - start ))
    
    num = len(fieldlines)
    for i in range(num):
        # Plot field line
        ax.plot( fieldlines[i][0,:], fieldlines[i][1,:], fieldlines[i][2,:], color='blue' )

def demo_trace_VTK():
    """Demo function to trace field line for a magnetic field stored in
    a VTK file.  

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """

    filename = "/tmp/3d__var_2_e20190902-041000-000.vtk"
    
    num = 10
    startPts = [None]*num
    x = np.linspace(-3,-10,num)
    for i in range(num) :
        startPts[i] = [x[i],0,0]
            
    # Setup plot for field lines
    ax = create_plot('Demo VTK file')    
    
    # Time process
    start = time.time()
    
    fl = mf.fieldlines_cartesian_unstructured_VTKfile(filename = filename,
                    Stop_Function = mf.trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest',
                    startPts = None,
                    direction = None,
                    F_field = 'b')
    
    fl.setstartpoints(startPts, mf.integrate_direction.both)
    
    # Setup multitrace for tracing field lines
    fieldlines = fl.tracefieldlines()
    
    print('Elapsed time:' + str( time.time() - start ))
    
    num = len(fieldlines)
    for i in range(num):
        # Plot field line
        ax.plot( fieldlines[i][0,:], fieldlines[i][1,:], fieldlines[i][2,:], color='blue' )

if __name__ == "__main__":
    demo_trace_BATSRUS()
    demo_trace_swmfio_BATSRUS()
    demo_trace_VTK()
