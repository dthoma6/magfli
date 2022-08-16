#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:52:10 2022

@author: dean
"""

import magfli as mf
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import pandas as pd
import pickle as pickle

# First part of filename for BATSRUS and VTK files
# i.e., VTK file is file + '.vtk'
Filebase = "/tmp/3d__var_2_e20190902-041000-000"
Tot_start_pts = 2

def add_array_to_df( df = None, label = None, arr = None ):
    """Add list of 2-D numpy arrays to pandas dataframe with each column 
    named label_1, label_2, etc.
    
    Inputs
    -------
    df = original dataframe to which arr will be added
    label = string to label dataframe columns
    arr = list of np.arrays
    
    Returns
    -------
    df = dataframe containing array arr.
    
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(arr, list)
    assert isinstance(label, str)

    logging.info( label + ' added to dataframe' )

    # Add each array in arr list to the dataframe
    # Our arrays are either 1xN or 3xN
    n = len(arr)
    d = arr[0].ndim
    if ( d > 1 ):
        w = arr[0].shape[0] 
        assert( w == 3 )

    for i in range(n):
        if ( d == 1 ):
            name = label + '_' + str(i)
            tmp_df = pd.DataFrame( arr[i].T, columns = [name] )
            df = pd.concat([df, tmp_df], axis=1)
        else:
            namex = label + '_' + str(i) + '_x'
            namey = label + '_' + str(i) + '_y'
            namez = label + '_' + str(i) + '_z'
            tmp_df = pd.DataFrame( arr[i].T, columns = [namex, namey, namez] )
            df = pd.concat([df, tmp_df], axis=1)
        
    # Return df with added arrays
    return df 


def get_fieldline_length(fieldlines = None):
    """Given a group of n fieldlines, determine the incremental length at 
    each point along the fieldlines.  That is the length at the first point
    in the field line is 0, the length at the second point is 0 + the distance 
    between the 1st and 2nd point, ..., the length at point n+1 is the total 
    length to point n plus the distance between points n and n+1.  Routine 
    returns n length arrays, one for each of the n fieldlines.
    
    Inputs
    -------
    fieldlines = list of fieldline 3-d arrays, nth array has the x,y,z points for 
            the nth fieldline.
    
    Returns
    -------
    lengths = list of n 1-d arrays with the incremental fieldline length.

    """
    logging.info('Path lengths... ' )
    
    n = len(fieldlines)
    lengths = [None]*n
    
    for i in range(n):
        m = np.shape(fieldlines[i])[1]
        lengths_m = np.zeros(m)
        
        for j in range(m-1):
            lengths_m[j+1] = lengths_m[j] + \
                np.linalg.norm(fieldlines[i][:,j+1] - fieldlines[i][:,j])
        
        lengths[i] = lengths_m
        
    return lengths
    
def get_path_diff(lengths1 = None, lengths2 = None, 
                  fieldline1 = None, fieldline2 = None):
    """Given fieldlines 1 and 2, along with the arc length of the path
    to each point on the two fieldlines, determine the scalar distance between
    two points at the same arc length.  Note, arc length is measured along the
    fieldline path.  If the two processes that generated the fieldlines are 
    identical, the scalar differences are zero.  
    
    Inputs
    -------
    lengths1 and lengths2 = arc length to each point in the fieldline arrays.  
            Arc length is measured along the field line.
    fieldline1 and fieldline2 = 2-d arrays, each array has the x,y,z points for 
            the applicable fieldline.
    
    
    Returns
    -------
    delta = np.array of scalar distances between the two fieldlines.  The n points
        are from the shorter of the two fieldlines.
    lengths = np.array of arc lengths used to calculate delta.
    """
    
    from scipy.interpolate import interp1d
        
    logging.info('Path differences... ' )

    # Use the shorter fieldline as the base for measuring the delta between
    # two field lines.  Measure delta from each point on shorter
    # fieldline to cooresponding point on longer fieldline.  By definition,
    # corresponding points are the same arc length from start point.
    if lengths1[-1] < lengths2[-1]: 
         # Create 1D interpolator to map the long fieldline points onto short
         # field line
         lengthinterp2 = interp1d(lengths2, fieldline2, fill_value="extrapolate")
         fieldline2NEW = lengthinterp2(lengths1)
         
         # Step through points on short fieldline and calculate distance beween
         # points
         n = np.shape(fieldline1)[1]
         delta = np.zeros(n)
         for m in range(n):
             delta[m] = np.linalg.norm(fieldline1[:,m] - fieldline2NEW[:,m])
        
         return delta, lengths1

    else:
         # Create 1D interpolator to map the long fieldline points onto short
         # field line
         lengthinterp1 = interp1d(lengths1, fieldline1, fill_value="extrapolate")
         fieldline1NEW = lengthinterp1(lengths2)
         
         # Step through points on short fieldline and calculate distance beween
         # points
         n = np.shape(fieldline2)[1]
         delta = np.zeros(n)
         for m in range(n):
             delta[m] = np.linalg.norm(fieldline2[:,m] - fieldline1NEW[:,m])

         return delta, lengths2
 
def get_mag_F(filename = None, fieldline = None, SWMFIO = True):
    """Use unstructured BATSRUS file call to get magnitude of F field 

    Inputs
    -------
    filename = BATSRUS file.
    fieldlines = 3-d arrays that has the x,y,z points for the fieldline.
    SWMFIO = use SWMFIO or SCIPY interpolator
    
    Returns
    -------
    magF = magnitude of F at each x,y,z point.
 
    """
    # Create swmfio BATSRUS fieldlines calculator, which provides interpolator
    # to estimate F field at any x,y,z point
    
    logging.info('Magnetic field magnitude... ' )

    if( SWMFIO ):
        fl = mf.fieldlines_cartesian_unstructured_swmfio_BATSRUSfile(filename = filename,
                        Stop_Function = mf.trace_stop_earth_box, 
                        tol = 1e-5, 
                        grid_spacing = 0.001, 
                        max_length = 100, 
                        method_ode = 'RK23',
                        method_interp = 'swmfio',
                        start_pts = None,
                        direction = None )
    else:
        fl = mf.fieldlines_cartesian_unstructured_BATSRUSfile(filename = filename,
                        Stop_Function = mf.trace_stop_earth_box, 
                        tol = 1e-5, 
                        grid_spacing = 0.001, 
                        max_length = 100, 
                        method_ode = 'RK23',
                        method_interp = 'nearest',
                        start_pts = None,
                        direction = None )
    
    # We calculate magnitude F along fieldline
    # Start       
    n = np.shape(fieldline)[1]
    magF = np.zeros(n)
    for m in range(n):
        magF[m] = np.linalg.norm(fl.field_value(fieldline[:,m]))

    return magF

def get_BATRSUS_measure(filename = Filebase, fieldline = None):
    """Find the BATSRUS measure at each point along a field line

    Inputs
    -------
    fieldline - 3-d arrays that has the x,y,z points for the fieldline.
    
    Returns
    -------
    measures - np.array of the measure at each x,y,z point.

    """
    # Read in BATSRUS file
    
    logging.info('BATSRUS measures... ' )

    import swmfio
    batsclass = swmfio.read_batsrus(filename)
       
    # Extract x, y, z and measure from file
    var_dict = dict(batsclass.varidx)
    x = batsclass.data_arr[:,var_dict['x']][:]
    y = batsclass.data_arr[:,var_dict['y']][:]
    z = batsclass.data_arr[:,var_dict['z']][:]
        
    m = batsclass.data_arr[:,var_dict['measure']][:]

    # Set up nearest neighbor interpolator so we can find 
    # the measure for each point x,y,z in the fieldline
    from scipy.interpolate import NearestNDInterpolator
    m_interpolate = NearestNDInterpolator(list(zip(x, y, z)), m )
    
    # Step through points on the fieldline and find measure at each point
    n = np.shape(fieldline)[1]
    measure = np.zeros(n)
    for m in range(n):
        measure[m] = m_interpolate( fieldline[:,m] )
       
    return measure

def get_scipy_BATSRUS(filename = None, delta = 0, method_interp = 'nearest'):
    """Use unstructured BATSRUS file call to get fieldlines 

    Inputs
    -------
    filename = BATSRUS file.
    
    Returns
    -------
    Elapsed time = total time to read and process file.
    BATSRUSlines = field lines 

    """
    # Create a set of start points for fieldlines
    num = Tot_start_pts
    start_pts = [None]*num
    x = np.linspace(-3+delta,-10+delta,num)
    for i in range(num) :
        start_pts[i] = [x[i],0,0]

    # Time process
    start_time = time.time()
    
    fl = mf.fieldlines_cartesian_unstructured_BATSRUSfile(filename = filename,
                    Stop_Function = mf.trace_stop_earth_box, 
                    tol = 1e-5, 
                    grid_spacing = 0.001, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = method_interp,
                    start_pts = None,
                    direction = None )
    
    fl.set_start_points(start_pts, mf.integrate_direction.both)
    
    read_time = time.time()

    # Setup multitrace for tracing field lines
    BATSRUSlines = fl.trace_field_lines()
    
    return (read_time - start_time), (time.time() - read_time), BATSRUSlines

def get_swmfio_BATSRUS(filename = None, delta = 0):
    """Use unstructured BATSRUS file call to get fieldlines 

    Inputs
    -------
    filename = BATSRUS file.
    
    Returns
    -------
    Elapsed time = total time to read and process file.
    BATSRUSlines = field lines 

    """
    # Create a set of start points for fieldlines
    num = Tot_start_pts
    start_pts = [None]*num
    x = np.linspace(-3+delta,-10+delta,num)
    for i in range(num) :
        start_pts[i] = [x[i],0,0]

    # Time process
    start_time = time.time()
    
    fl = mf.fieldlines_cartesian_unstructured_swmfio_BATSRUSfile(filename = filename,
                    Stop_Function = mf.trace_stop_earth_box, 
                    tol = 1e-5, 
                    grid_spacing = 0.001, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'swmfio',
                    start_pts = None,
                    direction = None )
    
    fl.set_start_points(start_pts, mf.integrate_direction.both)
    
    read_time = time.time()

    # Setup multitrace for tracing field lines
    BATSRUSlines = fl.trace_field_lines()
    
    return (read_time - start_time), (time.time() - read_time), BATSRUSlines
  
def get_scipy_VTK( filename = None, delta = 0, cell_centers = None ):
    """Use unstructured VTK file call to get fieldlines 

    Inputs
    -------
    filename = VTK file without .vtk extension.
    
    Returns
    -------
    Elapsed time = total time to read and process file.
    VTKlines = field lines 

    """
    # Create a set of start points for fieldlines
    num = Tot_start_pts
    start_pts = [None]*num
    x = np.linspace(-3+delta,-10+delta,num)
    for i in range(num) :
        start_pts[i] = [x[i],0,0]

    # Time process
    start_time = time.time()
    
    file = filename + '.vtk'
    fl = mf.fieldlines_cartesian_unstructured_VTKfile(filename = file,
                   Stop_Function = mf.trace_stop_earth_box, 
                   tol = 1e-5, 
                   grid_spacing = 0.001, 
                   max_length = 100, 
                   method_ode = 'RK23',
                   method_interp = 'nearest',
                   start_pts = None,
                   direction = None,
                   cell_centers = cell_centers)

    fl.set_start_points(start_pts, mf.integrate_direction.both)
    
    read_time = time.time()

    # Setup multitrace for tracing field lines
    VTKlines = fl.trace_field_lines()
    
    return (read_time - start_time), (time.time() - read_time), VTKlines


def get_paraview_VTK( filename = None, delta = 0 ):
    """Use unstructured Paraview VTK file call to get fieldlines 

    Inputs
    -------
    filename = VTK file without .vtk extension.
    
    Returns
    -------
    Elapsed time = total time to read and process file.
    VTKlines = field lines 

    """
    # Create a set of start points for fieldlines
    #seed_type = 'Line' 
    seed_type = 'Line' 
    start_pt1 = [-3+delta,0,0]
    # seed_type = 'Point Cloud' 
    # start_pt1 = [0,0,0]
    start_pt2 = [-10+delta,0,0]
    radius = 1
    resolution = Tot_start_pts - 1
 
    # Time process
    start_time = time.time()
    
    file = filename + '.vtk'
    fl = mf.fieldlines_cartesian_unstructured_paraview_VTKfile(filename = file,
                    max_length = 100, 
                    method_ode = 'RK2',
                    seed_type = seed_type,
                    start_pt1 = start_pt1,
                    start_pt2 = start_pt2,
                    radius = radius,
                    resolution = resolution,
                    direction = mf.integrate_direction.both,
                    F_field = 'b')
    
    read_time = time.time()
    
    # # Setup multitrace for tracing field lines
    VTKlines = fl.trace_field_lines()
    
    # Get rid of paraview items
    fl.reset()

    return (read_time - start_time), (time.time() - read_time), VTKlines    

def compare_methods_data():
    """Methods comparison function that determines the fieldlines using
    various methodologies and compares them.

    Inputs
    -------
    None.
    
    Returns
    -------
    None.

    """
    # Collect data (time to read files, time to calculate field lines, and
    # the field lines themselves) from the various methods 
         
    # Calculate baseline fieldlines and arc lengths
    
    Breadtime, Blinetime, Blines = get_scipy_BATSRUS( Filebase )
    Blengths = get_fieldline_length(Blines)
    
    BSreadtime, BSlinetime, BSlines = get_swmfio_BATSRUS( Filebase )
    BSlengths = get_fieldline_length(BSlines)
    
    Vreadtime, Vlinetime, Vlines = get_scipy_VTK( filename = Filebase )
    Vlengths = get_fieldline_length(Vlines)
    
    VCreadtime, VClinetime, VClines = get_scipy_VTK( filename = Filebase, 
            cell_centers='cc' )
    VClengths = get_fieldline_length(VClines)
    
    VPreadtime, VPlinetime, VPlines = get_paraview_VTK( filename = Filebase )
    VPlengths = get_fieldline_length(VPlines)
    
    # Calculate small pertubation on above fieldlines, shift origin by small
    # value, diff.  We will compare to above result to examine sensitivity to 
    # input point.
    
    diff = 0.005
    
    Breadtime_del, Blinetime_del, Blines_del = get_scipy_BATSRUS( Filebase, 
            delta = diff )
    Blengths_del = get_fieldline_length(Blines_del)
    
    BSreadtime_del, BSlinetime_del, BSlines_del = get_swmfio_BATSRUS( Filebase, 
            delta = diff )
    BSlengths_del = get_fieldline_length(BSlines_del)
       
    Vreadtime_del, Vlinetime_del, Vlines_del = get_scipy_VTK( filename = Filebase, 
            delta = diff )
    Vlengths_del = get_fieldline_length(Vlines_del)
    
    VCreadtime_del, VClinetime_del, VClines_del = get_scipy_VTK( filename = Filebase, 
            cell_centers='cc', delta = diff )
    VClengths_del = get_fieldline_length(VClines_del)
    
    VPreadtime_del, VPlinetime_del, VPlines_del = get_paraview_VTK( filename = Filebase, 
            delta = diff )
    VPlengths_del = get_fieldline_length(VPlines_del)
    
    # Get deltas between fieldlines from different methods.  Delta is defined
    # as the Euclidean distance between two points on two separate field lines
    # that are an arc distance x from the start point. Also get the magnitude 
    # of the field F along paths
     
    num = len(Blines) # Number of field lines in each group should be the same
    
    # Create empty storage for the results
    
    BVdelta = [None] * num
    BVCdelta = [None] * num
    BVPdelta = [None] * num
    BBSdelta = [None] * num
    
    BSVdelta = [None] * num
    BSVCdelta = [None] * num
    BSVPdelta = [None] * num
    
    VVCdelta = [None] * num
    VVPdelta = [None] * num
    
    BBdelta = [None] * num
    BSBSdelta = [None] * num
    VVdelta = [None] * num
    VCVCdelta = [None] * num
    VPVPdelta = [None] * num
    
    BVdelta_len = [None] * num
    BVCdelta_len = [None] * num
    BVPdelta_len = [None] * num
    BBSdelta_len = [None] * num
    
    BSVdelta_len = [None] * num
    BSVCdelta_len = [None] * num
    BSVPdelta_len = [None] * num
    
    VVCdelta_len = [None] * num
    VVPdelta_len = [None] * num
    
    BBdelta_len = [None] * num
    BSBSdelta_len = [None] * num
    VVdelta_len = [None] * num
    VCVCdelta_len = [None] * num
    VPVPdelta_len = [None] * num
    
    BmagB = [None] * num
    BSmagB = [None] * num
    
    Bmeasure = [None] * num
    
    # Calculate deltas, B field magnitude, and measures
    for i in range(num):
        BVdelta[i], BVdelta_len[i]= get_path_diff(lengths1 = Blengths[i], lengths2 = Vlengths[i], 
                          fieldline1 = Blines[i], fieldline2 = Vlines[i])
        
        BVCdelta[i], BVCdelta_len[i] = get_path_diff(lengths1 = Blengths[i], lengths2 = VClengths[i], 
                          fieldline1 = Blines[i], fieldline2 = VClines[i])
        
        BVPdelta[i], BVPdelta_len[i] = get_path_diff(lengths1 = Blengths[i], lengths2 = VPlengths[i], 
                          fieldline1 = Blines[i], fieldline2 = VPlines[i])
        
        BBSdelta[i], BBSdelta_len[i] = get_path_diff(lengths1 = Blengths[i], lengths2 = BSlengths[i], 
                          fieldline1 = Blines[i], fieldline2 = BSlines[i])
        
        BSVdelta[i], BSVdelta_len[i] = get_path_diff(lengths1 = BSlengths[i], lengths2 = Vlengths[i], 
                          fieldline1 = BSlines[i], fieldline2 = Vlines[i])
        
        BSVCdelta[i], BSVCdelta_len[i] = get_path_diff(lengths1 = BSlengths[i], lengths2 = VClengths[i], 
                          fieldline1 = BSlines[i], fieldline2 = VClines[i])
        
        BSVPdelta[i], BSVPdelta_len[i] = get_path_diff(lengths1 = BSlengths[i], lengths2 = VPlengths[i], 
                          fieldline1 = BSlines[i], fieldline2 = VPlines[i])
        
        VVCdelta[i], VVCdelta_len[i] = get_path_diff(lengths1 = Vlengths[i], lengths2 = VClengths[i], 
                          fieldline1 = Vlines[i], fieldline2 = VClines[i])

        VVPdelta[i], VVPdelta_len[i] = get_path_diff(lengths1 = Vlengths[i], lengths2 = VPlengths[i], 
                          fieldline1 = Vlines[i], fieldline2 = VPlines[i])

        BBdelta[i], BBdelta_len[i] = get_path_diff(lengths1 = Blengths[i], lengths2 = Blengths_del[i], 
                          fieldline1 = Blines[i], fieldline2 = Blines_del[i])
        
        BSBSdelta[i], BSBSdelta_len[i] = get_path_diff(lengths1 = BSlengths[i], lengths2 = BSlengths_del[i], 
                          fieldline1 = BSlines[i], fieldline2 = BSlines_del[i])
         
        VVdelta[i], VVdelta_len[i] = get_path_diff(lengths1 = Vlengths[i], lengths2 = Vlengths_del[i], 
                          fieldline1 = Vlines[i], fieldline2 = Vlines_del[i])
        
        VCVCdelta[i], VCVCdelta_len[i] = get_path_diff(lengths1 = VClengths[i], lengths2 = VClengths_del[i], 
                         fieldline1 = VClines[i], fieldline2 = VClines_del[i])
       
        VPVPdelta[i], VPVPdelta_len[i] = get_path_diff(lengths1 = VPlengths[i], lengths2 = VPlengths_del[i], 
                         fieldline1 = VPlines[i], fieldline2 = VPlines_del[i])
       
        BmagB[i] = get_mag_F(filename = Filebase, fieldline = Blines[i], SWMFIO = False)
     
        BSmagB[i] = get_mag_F(filename = Filebase, fieldline = BSlines[i], SWMFIO = True )
        
        Bmeasure[i] = get_BATRSUS_measure( filename = Filebase, fieldline = Blines[i] )
         
    # Create empty dataframe and add results from above to it, and store
    # dataframe in a file 
    
    results = pd.DataFrame()
       
    results = add_array_to_df( results, 'BATSRUS SCIPY Field Line', Blines )
    results = add_array_to_df( results, 'BATSRUS SCIPY Arc Length', Blengths )
    results = add_array_to_df( results, 'BATSRUS SWMFIO Field Line', BSlines )
    results = add_array_to_df( results, 'BATSRUS SWMFIO Arc Length', BSlengths )
    results = add_array_to_df( results, 'VTK SCIPY Field Line', Vlines )
    results = add_array_to_df( results, 'VTK SCIPY Arc Length', Vlengths )
    results = add_array_to_df( results, 'VTK SCIPY (CC) Field Line', VClines )
    results = add_array_to_df( results, 'VTK SCIPY (CC) Arc Length', VClengths )
    results = add_array_to_df( results, 'VTK PARA Field Line', VPlines )
    results = add_array_to_df( results, 'VTK PARA Arc Length', VPlengths )
    
    results = add_array_to_df( results, 'BATSRUS SCIPY Field Line Delta', Blines_del )
    results = add_array_to_df( results, 'BATSRUS SCIPY Arc Length Delta', Blengths_del )
    results = add_array_to_df( results, 'BATSRUS SWMFIO Field Line Delta', BSlines_del )
    results = add_array_to_df( results, 'BATSRUS SWMFIO Arc Length Delta', BSlengths_del )
    results = add_array_to_df( results, 'VTK SCIPY Field Line Delta', Vlines_del )
    results = add_array_to_df( results, 'VTK SCIPY Arc Length Delta', Vlengths_del )
    results = add_array_to_df( results, 'VTK SCIPY (CC) Field Line Delta', VClines_del )
    results = add_array_to_df( results, 'VTK SCIPY (CC) Arc Length Delta', VClengths_del )
    results = add_array_to_df( results, 'VTK PARA Field Line Delta', VPlines_del )
    results = add_array_to_df( results, 'VTK PARA Arc Length Delta', VPlengths_del )
        
    results = add_array_to_df( results, 'BATSRUS SCIPY VTK SCIPY Delta', BVdelta )
    results = add_array_to_df( results, 'BATSRUS SCIPY VTK SCIPY Delta Arc Lengths', BVdelta_len )
    results = add_array_to_df( results, 'BATSRUS SCIPY VTK SCIPY (CC) Delta', BVCdelta )
    results = add_array_to_df( results, 'BATSRUS SCIPY VTK SCIPY (CC) Delta Arc Lengths', BVCdelta_len )
    results = add_array_to_df( results, 'BATSRUS SCIPY VTK PARA Delta', BVPdelta )
    results = add_array_to_df( results, 'BATSRUS SCIPY VTK PARA Delta Arc Lengths', BVPdelta_len )
    results = add_array_to_df( results, 'BATSRUS SCIPY BATSRUS SWMFIO Delta', BBSdelta )
    results = add_array_to_df( results, 'BATSRUS SCIPY BATSRUS SWMFIO Delta Arc Lengths', BBSdelta_len )
    
    results = add_array_to_df( results, 'BATSRUS SWMFIO VTK SCIPY Delta', BSVdelta )
    results = add_array_to_df( results, 'BATSRUS SWMFIO VTK SCIPY Delta Arc Lengths', BSVdelta_len )
    results = add_array_to_df( results, 'BATSRUS SWMFIO VTK SCIPY (CC) Delta', BSVCdelta )
    results = add_array_to_df( results, 'BATSRUS SWMFIO VTK SCIPY (CC) Delta Arc Lengths', BSVCdelta_len )
    results = add_array_to_df( results, 'BATSRUS SWMFIO VTK PARA Delta', BSVPdelta )
    results = add_array_to_df( results, 'BATSRUS SWMFIO VTK PARA Delta Arc Lengths', BSVPdelta_len )
       
    results = add_array_to_df( results, 'VTK SCIPY VTK SCIPY (CC) Delta', VVCdelta )
    results = add_array_to_df( results, 'VTK SCIPY VTK SCIPY (CC) Delta Arc Lengths', VVCdelta_len )
    results = add_array_to_df( results, 'VTK SCIPY VTK PARA Delta', VVPdelta )       
    results = add_array_to_df( results, 'VTK SCIPY VTK PARA Delta Arc Lengths', VVPdelta_len )

    results = add_array_to_df( results, 'BATSRUS SCIPY (del) Field Line', Blines_del )
    results = add_array_to_df( results, 'BATSRUS SCIPY (del) Arc Length', Blengths_del )
    results = add_array_to_df( results, 'BATSRUS SWMFIO (del) Field Line', BSlines_del )
    results = add_array_to_df( results, 'BATSRUS SWMFIO (del) Arc Length', BSlengths_del )
    results = add_array_to_df( results, 'VTK PARA (del) Field Line', VPlines_del )
    results = add_array_to_df( results, 'VTK PARA (del) Arc Length', VPlengths_del )
 
    results = add_array_to_df( results, 'BATSRUS SCIPY BATSRUS SCIPY (del) Delta', BBdelta )
    results = add_array_to_df( results, 'BATSRUS SCIPY BATSRUS SCIPY (del) Delta Arc Lengths', BBdelta_len )
    results = add_array_to_df( results, 'BATSRUS SWMFIO BATSRUS SWMFIO (del) Delta', BSBSdelta )
    results = add_array_to_df( results, 'BATSRUS SWMFIO BATSRUS SWMFIO (del) Delta Arc Lengths', BSBSdelta_len )
    results = add_array_to_df( results, 'VTK SCIPY VTK SCIPY (del) Delta', VVdelta )
    results = add_array_to_df( results, 'VTK SCIPY VTK SCIPY (del) Delta Arc Lengths', VVdelta_len )
    results = add_array_to_df( results, 'VTK SCIPY (CC) VTK SCIPY (CC) (del) Delta', VCVCdelta )   
    results = add_array_to_df( results, 'VTK SCIPY (CC) VTK SCIPY (CC) (del) Delta Arc Lengths', VCVCdelta_len )   
    results = add_array_to_df( results, 'VTK PARA VTK PARA (del) Delta', VPVPdelta )
    results = add_array_to_df( results, 'VTK PARA VTK PARA (del) Delta Arc Lengths', VPVPdelta_len )
       
    results = add_array_to_df( results, 'BATSRUS SCIPY B Mag', BmagB )
    results = add_array_to_df( results, 'BATSRUS SWMFIO B Mag', BSmagB )
    results = add_array_to_df( results, 'BATSRUS SCIPY Measure', Bmeasure )
    
    # Defragment results (because I added a LOT of columns) by copying it
    results = results.copy()
    
    results.to_pickle('compare_methods.pkl')
    
    return

if __name__ == "__main__":
    compare_methods_data()
