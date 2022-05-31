#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:09:52 2022

@author: Dean Thomas
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
import types
import logging

def trace_stop_earth(s, X, xmin, ymin, zmin, xmax, ymax, zmax):
    """Event function used by solve_ivp.  We check to see if we're outside the
    earth, i.e., r > 1 in units of earth's radius, and terminate solve_ivp when
    r < 1
    
    Inputs:
        s = distance along field line
        X = current postion on field line
        xmin, ymin, zmin, xmax, ymax, zmax are unused
        
    Outputs:
        True/False - solve_ivp continues on True
    """

    r = np.linalg.norm(X)
    if( s == 0 ): return True
    return r > 1

def trace_stop_box(s, X, xmin, ymin, zmin, xmax, ymax, zmax):
    """Event function used by solve_ivp.  We check to see if we're inside the
    bounds of a rectangular box defined by xmin, ymin, zmin, xmax, ymax, and 
    zmax.  Terminate solve_ivp when we're outside the box.

    Inputs:
        s = distance along field line
        X = current postion on field line
        xmin, ymin, zmin, xmax, ymax, zmax = two corners of bounding box
        
    Outputs:
        True/False - solve_ivp continues on True
    """

    if( s == 0 ): return True
    if( X[0] < xmin or X[0] > xmax ): return False
    if( X[1] < ymin or X[1] > ymax ): return False
    if( X[2] < zmin or X[2] > zmax ): return False
    return True

def trace_stop_earth_box(s, X, xmin, ymin, zmin, xmax, ymax, zmax):
    """Event function used by solve_ivp.  We check to see if we're outside the
    earth, i.e., r > 1 in units of earth's radius, and whether we're inside the
    bounds of a rectangular box defined by xmin, ymin, zmin, xmax, ymax, and 
    zmax.  Terminate solve_ivp when we're inside the earth or outside the box.    
 
    Inputs:
        s = distance along field line
        X = current postion on field line
        xmin, ymin, zmin, xmax, ymax, zmax = two corners of bounding box

    Outputs:
        True/False - solve_ivp continues on True
    """

    if( s == 0 ): return True
    if( X[0] < xmin or X[0] > xmax ): return False
    if( X[1] < ymin or X[1] > ymax ): return False
    if( X[2] < zmin or X[2] > zmax ): return False
    r = np.linalg.norm(X)
    return r > 1

def trace_stop_none(s, X, xmin, ymin, zmin, xmax, ymax, zmax):
    """Event function used by solve_ivp.  No stopping, always continue.
 
    Inputs:
        s = distance along field line
        X = current postion on field line
        xmin, ymin, zmin, xmax, ymax, zmax are unused
        All inputs are ignored in this function

    Outputs:
        True/False - solve_ivp continues on True
    """
    return True

def trace_field_line_cartesian_function( X0, Xmin, Xmax, Field_Function, 
                               Stop_Function = trace_stop_earth, 
                               forward = True, tol = 1e-5, 
                               grid_spacing = 0.01, max_length = 100, 
                               method_ode = 'RK23' ):
    """Trace a magnetic field line based on start point XO, the initial x,y,z 
    position, and the provided Field function.  Algorithm uses solve_ivp to step
    along the field line
    
    Inputs:
        X0 = initial field position (x,y,z) for IVP
        Xmin = min x, min y, and min z defining one corner of box bounding domain
        Xmax = max x, max y, and max z defining second corner of bounding box
        Field_Function = function that provides B field vector at a specified X (x,y,z)
        Stop_Function = decides whether to end solve_ivp early
        forward = proceed forward along field line (+step) or backwards (-step)
        tol = tolerence for solve_ivp
        grid_spacing = grid spacing for solve_ivp
        max_length = maximum length (s) of field line
        method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
       
    Outputs:
        field_line.y = x,y,z points along field line starting at X0
    """
    assert isinstance(Field_Function, types.FunctionType)
    assert isinstance(Stop_Function, types.FunctionType)
    
    logging.info('Tracing field line...' + str(X0))

    # Function dXds is used by solve_ivp to solve ODE,
    # dX/ds = dB/|B|, that is used for tracing magnetic field lines
    # X (position) and B (magnetic field) are vectors.
    # s is the distance down the field line from initial point X0
    def dXds( s,X, xmin, ymin, zmin, xmax, ymax, zmax ):
        B = Field_Function( X )
        if( not forward ):
            B = np.negative( B )
        B_mag = np.linalg.norm(B)
        return B/B_mag
    
    # Tell solve_ivp that it should use the Stop_Function to terminate the 
    # integration.  Our stop functions made sure that we stay within the 
    # problem's domain.  See definitions of trace_stop_... above.
    Stop_Function.terminal = True
    
    # Define box that bounds the domain of the solution.
    # Xmin and Xmax are opposite corners of box.  The bounds
    # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
    # and zmax. (See trace_stop_... function definitions above. Some of
    # the trace_stop_... functions ignore these bounds.)
    bounds = Xmin + Xmax
    
    # Make grid for s from 0 to max_length, solve_ivp starts at 
    # s_grid[0] and runs until the end (unless aborted by Stop_Function).
    # s is the distance down the field line.
    s_grid = np.arange(0,max_length+grid_spacing,grid_spacing)

    # Call solve_ivp to find magnetic field line
    field_line = solve_ivp(fun=dXds, t_span=[s_grid[0], s_grid[-1]], 
                    y0=X0, t_eval=s_grid, rtol=tol, 
                    events=Stop_Function, method=method_ode,
                    args = bounds)
    
    # Check for error
    assert( field_line.status != -1 )
    
    return field_line.y
    
def trace_field_line_cartesian_regular_grid( X0, x, y, z, Bx, By, Bz,
                               Stop_Function = trace_stop_earth, 
                               forward = True, tol = 1e-5, 
                               grid_spacing = 0.01, max_length = 100, 
                               method_ode = 'RK23',
                               method_interp = 'nearest'):
    """Trace a magnetic field line based on start point XO, the initial x,y,z 
    position, and a regular grid with information on the magnetic field.  The 
    grid must be a regular grid; but the grid spacing however may be uneven.
    Algorithm uses solve_ivp to step along the field line
    
    Inputs:
        x, y, z = x,y,z each is a range of values defining the positions at 
            which the magnetic field is known.  x, y, and z vectors define
            extent of grid along each axis.
        Bx, By, Bz = value of magnetic field vector at each x,y,z
        Stop_Function = decides whether to end solve_ivp early
        forward = proceed forward along field line (+step) or backwards (-step)
        tol = tolerence for solve_ivp
        grid_spacing = grid spacing for solve_ivp
        max_length = maximum length (s) of field line
        method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
        method_interp = interpolator method, 'linear' or 'nearest'
        
    Outputs:
        field_line.y = x,y,z points along field line starting at X0
    """
    assert isinstance(Stop_Function, types.FunctionType)
    assert(method_interp == 'linear' or method_interp == 'nearest')
   
    logging.info('Tracing field line...' + str(X0))
    
    # Create interpolators for regular grid
    Bx_interpolate = RegularGridInterpolator((x, y, z), Bx, method=method_interp)
    By_interpolate = RegularGridInterpolator((x, y, z), By, method=method_interp)
    Bz_interpolate = RegularGridInterpolator((x, y, z), Bz, method=method_interp)
    
    # Function dXds is used by solve_ivp to solve ODE,
    # dX/ds = dB/|B|, that is used for tracing magnetic field lines
    # X (position) and B (magnetic field) are vectors.
    # s is the distance down the field line from initial point X0
    def dXds( s,X, xmin, ymin, zmin, xmax, ymax, zmax ):
        B = np.zeros(3)
        B[0] = Bx_interpolate(X)
        B[1] = By_interpolate(X)
        B[2] = Bz_interpolate(X)
        if( not forward ):
            B = np.negative( B )
        B_mag = np.linalg.norm(B)
        return B/B_mag
    
    # Tell solve_ivp that it should use the Stop_Function to terminate the 
    # integration.  Our stop functions made sure that we stay within the 
    # problem's domain.  See definitions of trace_stop_... above.
    Stop_Function.terminal = True
    
    # Define box that bounds the domain of the solution.
    # min and max are opposite corners of box.  The bounds
    # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
    # and zmax. (See trace_stop_... function definitions above. Some of
    # the trace_stop_... functions ignore these bounds.)
    bounds = [min(x), min(y), min(z)] + [max(x), max(y), max(z)]
    
    # Make grid for s from 0 to max_length, solve_ivp starts at 
    # s_grid[0] and runs until the end (unless aborted by Stop_Function).
    # s is the distance down the field line.
    s_grid = np.arange(0,max_length+grid_spacing,grid_spacing)

    # Call solve_ivp to find magnetic field line
    field_line = solve_ivp(fun=dXds, t_span=[s_grid[0], s_grid[-1]], 
                    y0=X0, t_eval=s_grid, rtol=tol, 
                    events=Stop_Function, method=method_ode,
                    args = bounds)
    
    # Check for error
    assert( field_line.status != -1 )
    
    return field_line.y
    
def trace_field_line_cartesian_unstructured( X0, x, y, z, Bx, By, Bz,
                               Stop_Function = trace_stop_earth, 
                               forward = True, tol = 1e-5, 
                               grid_spacing = 0.01, max_length = 100, 
                               method_ode = 'RK23',
                               method_interp = 'nearest'):
    """Trace a magnetic field line based on start point XO, the initial x,y,z 
    position, and the provided unstructured data on the magnetic field.  
    Algorithm uses solve_ivp to step along the field line
    
    Inputs:
        x, y, z = x,y,z each is a range of values defining the positions at 
            which the magnetic field is known
        Bx, By, Bz = value of magnetic field vector at each x,y,z
        Stop_Function = decides whether to end solve_ivp early
        forward = proceed forward along field line (+step) or backwards (-step)
        tol = tolerence for solve_ivp
        grid_spacing = grid spacing for solve_ivp
        max_length = maximum length (s) of field line
        method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
        method_interp = interpolator method, 'linear' or 'nearest'
        
    Outputs:
        field_line.y = x,y,z points along field line starting at X0
    """
    assert isinstance(Stop_Function, types.FunctionType)
    assert( method_interp == 'linear' or method_interp == 'nearest')
   
    logging.info('Tracing field line...' + str(X0))
    
    # Create interpolators for unstructured data
    # Use list(zip(...)) to convert x,y,z's => (x0,y0,z0), (x1,y1,z1), ...
    if( method_interp == 'linear'):
        Bx_interpolate = LinearNDInterpolator(list(zip(x, y, z)), Bx )
        By_interpolate = LinearNDInterpolator(list(zip(x, y, z)), By )
        Bz_interpolate = LinearNDInterpolator(list(zip(x, y, z)), Bz )
    if( method_interp == 'nearest'):
        Bx_interpolate = NearestNDInterpolator(list(zip(x, y, z)), Bx )
        By_interpolate = NearestNDInterpolator(list(zip(x, y, z)), By )
        Bz_interpolate = NearestNDInterpolator(list(zip(x, y, z)), Bz )
    
    # Function dXds is used by solve_ivp to solve ODE,
    # dX/ds = dB/|B|, that is used for tracing magnetic field lines
    # X (position) and B (magnetic field) are vectors.
    # s is the distance down the field line from initial point X0
    def dXds( s,X, xmin, ymin, zmin, xmax, ymax, zmax ):
        B = np.zeros(3)
        B[0] = Bx_interpolate(X)
        B[1] = By_interpolate(X)
        B[2] = Bz_interpolate(X)
        if( not forward ):
            B = np.negative( B )
        B_mag = np.linalg.norm(B)
        return B/B_mag
    
    # Tell solve_ivp that it should use the Stop_Function to terminate the 
    # integration.  Our stop functions made sure that we stay within the 
    # problem's domain.  See definitions of trace_stop_... above.
    Stop_Function.terminal = True
    
    # Define box that bounds the domain of the solution.
    # min and max are opposite corners of box.  The bounds
    # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
    # and zmax. (See trace_stop_... function definitions above. Some of
    # the trace_stop_... functions ignore these bounds.)
    bounds = [min(x), min(y), min(z)] + [max(x), max(y), max(z)]
    
    # Make grid for s from 0 to max_length, solve_ivp starts at 
    # s_grid[0] and runs until the end (unless aborted by Stop_Function).
    # s is the distance down the field line.
    s_grid = np.arange(0,max_length+grid_spacing,grid_spacing)

    # Call solve_ivp to find magnetic field line
    field_line = solve_ivp(fun=dXds, t_span=[s_grid[0], s_grid[-1]], 
                    y0=X0, t_eval=s_grid, rtol=tol, 
                    events=Stop_Function, method=method_ode,
                    args = bounds)
    
    # Check for error
    assert( field_line.status != -1 )
    
    return field_line.y
    
    