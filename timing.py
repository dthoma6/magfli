#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:53:54 2022

@author: Dean Thomas
"""

import magfli as mf
import numpy as np
import pandas as pd
import time

def timing_vary_unstructured_params(num_pts=100000, tolerance=1e-5,
                                    grid=0.1,
                                    method_o='RK23', 
                                    method_i='linear'):

    """Routine used to evaluate how long it takes to trace a magnetic field
    line as a function of various parameters.

    Inputs
    -------
    num_pts = number of points in unstructured random grid
    tolerance = tolerence specified to solve_ivp
    grid = distance between points on the field line
    method_ode = which method, e.g., 'RK23', used in solve_ivp 
    method_interp = whether interpolation is 'linear' or 'nearest'
    
    Returns
    -------
    elapsed = Elapsed clock time in seconds
    rms = RMS difference from analytic solution

    """
    # Point where trace begins    
    X0 = [ 1/2, 1/2, 1/np.sqrt(2) ]
    
    # Get the unstructured grid defining the magnetic field
    x, y, z, Bx, By, Bz = mf.dipole_earth_cartesian_unstructured([5,5,5],num_pts)
    
    start = time.time()
    
    # Setup multitrace
    # Requires trace be outside the earth (see mf.trace_stop_earth)
    mt = mf.multitrace_cartesian_unstructured( x, y, z, Bx, By, Bz,
                                   Stop_Function = mf.trace_stop_earth, 
                                   tol = tolerance, 
                                   grid_spacing = grid, 
                                   max_length = 5, 
                                   method_ode = method_o, 
                                   method_interp = method_i )
    
    # Trace field line
    field_line = mt.trace_field_line( X0, False )
    
    elapsed = time.time() - start

    # Calculate RMS difference from analytic solution
    shape = np.shape(field_line)
    diffs = np.zeros(shape[1])
    for i in range(shape[1]):
        diffs[i] = mf.dipole_mag_line_delta_cartesian(field_line[:,i], X0)

    rms = np.linalg.norm( diffs )/np.sqrt(len(diffs))
    
    return elapsed, rms


def timing_trace_unstructured():
    """Timing function that determines elapsed time to trace one field line.
    Elapsed time is for a simple dipole model of earth's magnetic field 
    specified in an unstructured set of points. The function will vary the 
    parameters of the tracing function to determine how elapsed time changes.

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    
    # Create storage for timing results
    cols = ['elapsed time (sec)', 'RMS', 'log(number points)', 'log(tolerance)', 'log(grid size)', 
           'method ode', 'method interpolation']
    results = pd.DataFrame(columns=cols)
    
    ode_methods = ['RK23', 'RK45', 'DOP853']
    interp_methods = ['linear', 'nearest']
    

    for pts in range(4,7):
        for tol in range(2,6):
            for grid_space in range(1,3):
                for meth_ode in ode_methods:
                    for meth_interp in interp_methods:
                        print( 'Processing...', pts, tol, grid_space, meth_ode, meth_interp )
                        elapsed, rms = timing_vary_unstructured_params(num_pts=10**pts, 
                                            tolerance=10**(-tol),
                                            grid=10**(-grid_space),
                                            method_o=meth_ode, 
                                            method_i=meth_interp)
                        
                        results.loc[len(results.index)] = [elapsed, rms, pts, -tol, -grid_space, 
                                                                 meth_ode, meth_interp]

                        
        
    results.to_pickle('timing.pkl')
    return

def timing_trace_plots():
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
    # Read results from timing_trace_unstructured()
    results = pd.read_pickle("timing.pkl")
    
    # Add a few columns based on the data in the dataframe
    results['tolerance'] = 10.0**results['log(tolerance)']
    results['number points'] = 10.0**results['log(number points)']
    results['grid size'] = 10.0**results['log(grid size)']
    
    # Plot results to identify patterns
    colors = {'linear':'red', 'nearest':'blue'}
    results.plot('number points','elapsed time (sec)',kind='scatter', logy=True, logx=True, 
                 color=results['method interpolation'].map(colors))
    results.plot('number points','RMS',kind='scatter', logy=True, logx=True,
                 color=results['method interpolation'].map(colors))
    results.plot('tolerance','elapsed time (sec)',kind='scatter', logy=True, logx=True,
                 color=results['method interpolation'].map(colors))
    results.plot('tolerance','RMS',kind='scatter', logy=True, logx=True,
                 color=results['method interpolation'].map(colors))
    results.plot('grid size','elapsed time (sec)',kind='scatter', logy=True, logx=True,
                 color=results['method interpolation'].map(colors))
    results.plot('grid size','RMS',kind='scatter', logy=True, logx=True,
                 color=results['method interpolation'].map(colors))
    results.plot('method ode','elapsed time (sec)',kind='scatter', logy=True,
                 color=results['method interpolation'].map(colors))
    results.plot('method ode','RMS',kind='scatter', logy=True, 
                 color=results['method interpolation'].map(colors))
    results.plot('method interpolation','elapsed time (sec)',kind='scatter', logy=True)
    results.plot('method interpolation','RMS',kind='scatter', logy=True)

    return

if __name__ == "__main__":
    #timing_trace_unstructured()
    timing_trace_plots()
