#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:29:49 2022

@author: Dean Thomas
"""

from magfli.trace import trace_stop_earth
from magfli.dipole import dipole_earth_cartesian
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
import numpy as np
import types
import logging


class multitrace_cartesian_function():
    """Trace multiple magnetic field lines through the provided magnetic field.
    The magnetic field is defined through a function that calculates the magnetic 
    field (Bx,By,Bz) at point (x,y,z).  Algorithm uses solve_ivp to step along
    the field line.
    """

    def __init__(self, Xmin, Xmax, 
                    Field_Function = dipole_earth_cartesian,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23' ):
        """Initialize multitrace_cartesian_function class
        
        Inputs:
            Xmin = min x, min y, and min z defining one corner of box bounding domain
            Xmax = max x, max y, and max z defining second corner of bounding box
            Field_Function = function that determines B field vector at a specified X (x,y,z)
            Stop_Function = function decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            
        Outputs:
            None.
        """

        logging.info('Initializing multitrace for function: ' + str(Xmax) + ' ' + 
                     str(Xmin) + ' ' + str(tol) + ' ' + str(grid_spacing) +
                     ' ' + str(max_length) + ' ' + method_ode)

        # Input error checks
        # assert isinstance(Xmin, types.ndarray)
        # assert isinstance(Xmax, types.ndarray)
        assert isinstance(Field_Function, types.FunctionType)
        assert isinstance(Stop_Function, types.FunctionType)
        assert( method_ode == 'RK23' or method_ode == 'RK45' or method_ode == 'DOP853')
        
        # Store instance data
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.Field_Function = Field_Function
        self.Stop_Function = Stop_Function
        self.tol = tol
        self.grid_spacing = grid_spacing
        self.max_length = max_length
        self.method_ode = method_ode

        # Tell solve_ivp that it should use the Stop_Function to terminate the 
        # integration.  Our stop functions makes sure that we stay within the 
        # problem's domain.  See definitions of trace_stop_... in trace.py.
        self.Stop_Function.terminal = True
        
        # Define box that bounds the domain of the solution.
        # Xmin and Xmax are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace.py. Some
        # of the trace_stop_... functions ignore these bounds.)
        self.bounds = Xmin + Xmax
        
        # Make grid for s from 0 to max_length, solve_ivp starts at 
        # s_grid[0] and runs until the end (unless aborted by Stop_Function).
        # s is the distance down the field line.
        self.s_grid = np.arange(0,max_length+grid_spacing,grid_spacing)
    
        return
    
    def trace_field_line(self, X0, forward):
        """Trace a magnetic field line based on start point XO and the provided 
        magnetic field. The algorithm uses solve_ivp to step along the field line
        
        Inputs:
            X0 = point at which to start the field line integration
            forward = proceed forward along field line (+step) or backwards (-step)
            
        Outputs:
            field_line.y = x,y,z points along field line starting at X0
        """
       
        logging.info('Tracing field line...' + str(X0))
    
        # Function dXds is used by solve_ivp to solve the ODE,
        # dX/ds = dB/|B|, which is used for tracing magnetic field lines.
        # X (position) and B (magnetic field) are vectors.
        # s is the distance down the field line from initial point X0
        def dXds( s,X, xmin, ymin, zmin, xmax, ymax, zmax ):
            B = self.Field_Function( X )
            if( not forward ):
                B = np.negative( B )
            B_mag = np.linalg.norm(B)
            return B/B_mag
        
        # Call solve_ivp to find magnetic field line
        field_line = solve_ivp(fun=dXds, t_span=[self.s_grid[0], self.s_grid[-1]], 
                        y0=X0, t_eval=self.s_grid, rtol=self.tol, 
                        events=self.Stop_Function, method=self.method_ode,
                        args = self.bounds)
        
        # Check for error
        assert( field_line.status != -1 )
        
        return field_line.y

    def trace_magnetic_field(self, X):
        """Use the field function to calculate the magnetic field at point X.
         
        Inputs:
            X = point at which to evaluate magnetic field integration
             
        Outputs:
            B[] = x,y,z components of magnetic field at point X
        """

        B = self.Field_Function( X )
        
        return B
        
class multitrace_cartesian_regular_grid():
    """Trace multiple magnetic field lines through the provided magnetic field.
    The magnetic field is defined through a regular grid (array) of points 
    (x,y,z) at which the magnetic field (Bx,By,Bz) is provided.  The locations of
    the x,y,z points are regularly spaced.  Algorithm uses solve_ivp to step along 
    the field line.
    """
    
    def __init__(self, x, y, z, Bx, By, Bz,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest' ):
        """Initialize multitrace_cartesian_regular_grid()
        
        Inputs:
            x, y, z = x,y,z each is a range of values defining the positions at 
                which the magnetic field is known.  x, y, and z vectors define
                extent of grid along each axis.
            Bx, By, Bz = value of magnetic field vector at each x,y,z
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            
        Outputs:
            None
        """
 
        logging.info('Initializing multitrace for regular grid: ' + str(tol) + 
                     ' ' + str(grid_spacing) + ' ' + str(max_length) + ' ' + 
                     method_ode + ' ' + method_interp)

        # Input error checks
        # assert isinstance(x, types.ndarray)
        # assert isinstance(y, types.ndarray)
        # assert isinstance(z, types.ndarray)
        # assert isinstance(Bx, types.ndarray)
        # assert isinstance(By, types.ndarray)
        # assert isinstance(Bz, types.ndarray)
        assert isinstance(Stop_Function, types.FunctionType)
        assert( method_ode == 'RK23' or method_ode == 'RK45' or method_ode == 'DOP853')
        assert( method_interp == 'linear' or method_interp == 'nearest')
        
        # Store instance data
        self.x = x
        self.y = y
        self.z = z
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.Stop_Function = Stop_Function
        self.tol = tol
        self.grid_spacing = grid_spacing
        self.max_length = max_length
        self.method_ode = method_ode
        self.method_interp = method_interp
 
        # Create interpolators for regular grid
        self.Bx_interpolate = RegularGridInterpolator((x, y, z), Bx, method=method_interp)
        self.By_interpolate = RegularGridInterpolator((x, y, z), By, method=method_interp)
        self.Bz_interpolate = RegularGridInterpolator((x, y, z), Bz, method=method_interp)
        
        # Tell solve_ivp that it should use the Stop_Function to terminate the 
        # integration.  Our stop functions made sure that we stay within the 
        # problem's domain.  See definitions of trace_stop_... in trace.py.
        self.Stop_Function.terminal = True
        
        # Define box that bounds the domain of the solution.
        # min and max are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace.py. Some
        # of the trace_stop_... functions ignore these bounds.)
        self.bounds = [min(x), min(y), min(z)] + [max(x), max(y), max(z)]
        
        # Make grid for s from 0 to max_length, solve_ivp starts at 
        # s_grid[0] and runs until the end (unless aborted by Stop_Function).
        # s is the distance down the field line.
        self.s_grid = np.arange(0,max_length+grid_spacing,grid_spacing)
        
        return
    
    def trace_field_line(self, X0, forward):
        """Trace a magnetic field line based on start point XO and the provided 
        magnetic field.  The magnetic field is defined by an regular grid of points 
        (x,y,z) at which the magnetic field (Bx,By,Bz) is provided.  The algorithm 
        uses solve_ivp to step along the field line
        
        Inputs:
            X0 = point at which to start the field line integration
            forward = proceed forward along field line (+step) or backwards (-step)
            
        Outputs:
            field_line.y = x,y,z points along field line starting at X0
        """
       
        logging.info('Tracing field line...' + str(X0))
                
        # Function dXds is used by solve_ivp to solve ODE,
        # dX/ds = dB/|B|, which is used for tracing magnetic field lines.
        # X (position) and B (magnetic field) are vectors.
        # s is the distance down the field line from initial point X0
        def dXds( s,X, xmin, ymin, zmin, xmax, ymax, zmax ):
            B = np.zeros(3)
            B[0] = self.Bx_interpolate(X)
            B[1] = self.By_interpolate(X)
            B[2] = self.Bz_interpolate(X)
            if( not forward ):
                B = np.negative( B )
            B_mag = np.linalg.norm(B)
            return B/B_mag
        
        # Call solve_ivp to find magnetic field line
        field_line = solve_ivp(fun=dXds, t_span=[self.s_grid[0], self.s_grid[-1]], 
                        y0=X0, t_eval=self.s_grid, rtol=self.tol, 
                        events=self.Stop_Function, method=self.method_ode,
                        args = self.bounds)
        
        # Check for error
        assert( field_line.status != -1 )
        
        return field_line.y
    
    def trace_magnetic_field(self, X):
        """Use the interpolator to estimate the magnetic field at point X.
         
        Inputs:
            X = point at which to evaluate magnetic field integration
             
        Outputs:
            B[] = x,y,z components of magnetic field at point X
        """

        B = np.zeros(3)
        B[0] = self.Bx_interpolate(X)
        B[1] = self.By_interpolate(X)
        B[2] = self.Bz_interpolate(X)
        
        return B

class multitrace_cartesian_unstructured():
    """Trace multiple magnetic field lines through the provided magnetic field.
    The magnetic field is defined through unstructured data, an array of points 
    (x,y,z) at which the magnetic field (Bx,By,Bz) is provided.  The locations of
    the x,y,z points is unconstrained.  Algorithm uses solve_ivp to step along 
    the field line
    """

    def __init__(self, x, y, z, Bx, By, Bz,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest' ):
 
        """Initialize multitrace_cartesian_unstructured
            
        Inputs:
            x, y, z = define the positions at which the magnetic field is known 
            Bx, By, Bz = value of magnetic field vector at each x,y,z
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            
        Outputs:
            None
        """
        logging.info('Initializing multitrace for unstructured data: ' + str(tol) + 
                     ' ' + str(grid_spacing) + ' ' + str(max_length) + ' ' + 
                     method_ode + ' ' + method_interp)

        # Input error checks
        # assert isinstance(x, types.ndarray)
        # assert isinstance(y, types.ndarray)
        # assert isinstance(z, types.ndarray)
        # assert isinstance(Bx, types.ndarray)
        # assert isinstance(By, types.ndarray)
        # assert isinstance(Bz, types.ndarray)
        assert isinstance(Stop_Function, types.FunctionType)
        assert( method_ode == 'RK23' or method_ode == 'RK45' or method_ode == 'DOP853')
        assert( method_interp == 'linear' or method_interp == 'nearest')

        # Store instance data
        self.x = x
        self.y = y
        self.z = z
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.Stop_Function = Stop_Function
        self.tol = tol
        self.grid_spacing = grid_spacing
        self.max_length = max_length
        self.method_ode = method_ode
        self.method_interp = method_interp
        
        # Create interpolators for unstructured data
        # Use list(zip(...)) to convert x,y,z's => (x0,y0,z0), (x1,y1,z1), ...
        if( method_interp == 'linear'):
            self.Bx_interpolate = LinearNDInterpolator(list(zip(x, y, z)), Bx )
            self.By_interpolate = LinearNDInterpolator(list(zip(x, y, z)), By )
            self.Bz_interpolate = LinearNDInterpolator(list(zip(x, y, z)), Bz )
        if( method_interp == 'nearest'):
            self.Bx_interpolate = NearestNDInterpolator(list(zip(x, y, z)), Bx )
            self.By_interpolate = NearestNDInterpolator(list(zip(x, y, z)), By )
            self.Bz_interpolate = NearestNDInterpolator(list(zip(x, y, z)), Bz )
 
        # Tell solve_ivp that it should use the Stop_Function to terminate the 
        # integration.  Our stop functions made sure that we stay within the 
        # problem's domain.  See definitions of trace_stop_... in trace.py.
        self.Stop_Function.terminal = True
        
        # Define box that bounds the domain of the solution.
        # min and max are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace.py. Some
        # of the trace_stop_... functions ignore these bounds.)
        self.bounds = [min(x), min(y), min(z)] + [max(x), max(y), max(z)]
        
        # Make grid for s from 0 to max_length, solve_ivp starts at 
        # s_grid[0] and runs until the end (unless aborted by Stop_Function).
        # s is the distance down the field line.
        self.s_grid = np.arange(0,max_length+grid_spacing,grid_spacing)

        return

    def trace_field_line(self, X0, forward):
        """Trace a magnetic field line based on start point XO and the provided 
        magnetic field.  The magnetic field is defined by an array of points 
        (x,y,z) at which the magnetic field (Bx,By,Bz) is provided.  The algorithm 
        uses solve_ivp to step along the field line
        
        Inputs:
            X0 = point at which to start the field line integration
            forward = proceed forward along field line (+step) or backwards (-step)
            
        Outputs:
            field_line.y = x,y,z points along field line starting at X0
        """
       
        logging.info('Tracing field line...' + str(X0))
                
        # Function dXds is used by solve_ivp to solve ODE,
        # dX/ds = dB/|B|, which is used for tracing magnetic field lines.
        # X (position) and B (magnetic field) are vectors.
        # s is the distance down the field line from initial point X0
        def dXds( s,X, xmin, ymin, zmin, xmax, ymax, zmax ):
            B = np.zeros(3)
            B[0] = self.Bx_interpolate(X)
            B[1] = self.By_interpolate(X)
            B[2] = self.Bz_interpolate(X)
            if( not forward ):
                B = np.negative( B )
            B_mag = np.linalg.norm(B)
            return B/B_mag
        
        # Call solve_ivp to find magnetic field line
        field_line = solve_ivp(fun=dXds, t_span=[self.s_grid[0], self.s_grid[-1]], 
                        y0=X0, t_eval=self.s_grid, rtol=self.tol, 
                        events=self.Stop_Function, method=self.method_ode,
                        args = self.bounds)
        
        # Check for error
        assert( field_line.status != -1 )
        
        return field_line.y
        
    def trace_magnetic_field(self, X):
        """Use the interpolator to estimate the magnetic field at point X.
         
        Inputs:
            X = point at which to evaluate magnetic field integration
             
        Outputs:
            B[] = x,y,z components of magnetic field at point X
        """

        B = np.zeros(3)
        B[0] = self.Bx_interpolate(X)
        B[1] = self.By_interpolate(X)
        B[2] = self.Bz_interpolate(X)
        
        return B
       