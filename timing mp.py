#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:18:20 2022

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
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    
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
    
def timing_data():
    filename = "/tmp/3d__var_2_e20190902-041000-000"
     
    # Extract data from BATSRUS file
    # Read BATSRUS file
    import swmfio
    batsrus = swmfio.read_batsrus(filename)
    assert( batsrus != None )
       
    # Extract x, y, z and Fx, Fy, Fz from file
    var_dict = dict(batsrus.varidx)
    
    x_field = 'x'
    y_field = 'y' 
    z_field = 'z'
    Fx_field = 'bx' 
    Fy_field = 'by' 
    Fz_field = 'bz'
    
    x = batsrus.data_arr[:,var_dict[x_field]][:]
    y = batsrus.data_arr[:,var_dict[y_field]][:]
    z = batsrus.data_arr[:,var_dict[z_field]][:]
        
    Fx = batsrus.data_arr[:,var_dict[Fx_field]][:]
    Fy = batsrus.data_arr[:,var_dict[Fy_field]][:]
    Fz = batsrus.data_arr[:,var_dict[Fz_field]][:]
    
    # Create a set of start points for fieldlines
    start_pts = get_start_points()

    # methods = ('RK23', 'RK45', 'DOP853','BDF', 'LSODA' )
    methods = ('RK23', 'RK45', 'DOP853' )
    approaches = ('serial', 'parallel')

    # Create storage for timing results
    import pandas as pd
    cols = ['elapsed time (sec)', 'Method ODE', 'Approach' ]
    results = pd.DataFrame(columns=cols)
    
    for approach in approaches:
        for method in methods:
            fl = mf.fieldlines_cartesian_unstructured( x = x, y = y, z = z,
                                Fx = Fx, Fy = Fy, Fz = Fz, 
                                Stop_Function = mf.trace_stop_earth, 
                                tol = 1e-5, 
                                grid_spacing = 0.01, 
                                max_length = 100, 
                                method_ode = method,
                                method_interp = 'nearest',
                                start_pts = start_pts,
                                direction = mf.integrate_direction.both)
                
            # Time process
            start = time.time()
            
            try:
                # Setup multitrace for tracing field lines
                if approach == 'serial':
                    fieldlines = fl.trace_field_lines()
                else:
                    fieldlines = fl.trace_mp_field_lines()
                
                elapsed = time.time() - start

                # Setup plot for field lines
                title = method + ' ' + approach
                ax = create_plot(title)    
                
                # Plot fieldlines    
                num = len(fieldlines)
                for i in range(num):
                    # Plot field line
                        ax.plot( fieldlines[i][0,:], fieldlines[i][1,:], fieldlines[i][2,:], color='blue' )
            except BaseException:
                elapsed = float('NaN')
                
            results.loc[len(results.index)] = [elapsed, method, approach]
            
    results.to_pickle('timing mp.pkl')
    return

def timing_plots():
    """Timing function that creates plots to see if there is any pattern
    in elapsed time or RMS as a function of various parameters.  See 
    timing_trace_unstructured for details on the parameters explored.

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Read results 
    results = pd.read_pickle("timing mp.pkl")
    
     
    # Plot results to identify patterns
    colors = {'serial':'red', 'parallel':'blue'}
    results.plot('Method ODE','elapsed time (sec)', 
                  kind='scatter', 
                  legend=True, 
                  color=results['Approach'].map(colors))
    plt.legend(['Serial', 'Parallel'])
    return


if __name__ == "__main__":
    # timing_data()
    timing_plots()
