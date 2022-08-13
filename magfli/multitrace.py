#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:29:49 2022

@author: Dean Thomas
"""

from magfli.trace_stop_funcs import trace_stop_earth
from magfli.dipole import dipole_earth_cartesian
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
import numpy as np
import types
import logging

class multitrace_cartesian_grid_base():
    """Base class for tracing multiple field lines through the provided field. 
    The field is defined through a grid (array) of points (x,y,z) at which the 
    field (Fx,Fy,Fz) is provided.  Algorithm uses solve_ivp to step along the 
    field line. Most changes for child classes will be in __init__().  
    trace_field_line() and trace_field_value() are common to child classes
    using scipy interpolators.  Classes with other interpolators will need to
    implement everything.
    """
    
    def __init__(self, Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest' ):
        """Initialize multitrace_cartesian_grid_base()
        
        Inputs:
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            
        Outputs:
            None
        """
 
        logging.info('Initializing multitrace base class: ' + str(tol) + 
                     ' ' + str(grid_spacing) + ' ' + str(max_length) + ' ' + 
                     method_ode + ' ' + method_interp)

        # Input error checks
        assert isinstance(Stop_Function, types.FunctionType)
        assert( method_ode == 'RK23' or method_ode == 'RK45' or method_ode == 'DOP853')
        assert( method_interp == 'linear' or method_interp == 'nearest')
        
        # Store instance data
        self.Stop_Function = Stop_Function
        self.tol = tol
        self.grid_spacing = grid_spacing
        self.max_length = max_length
        self.method_ode = method_ode
        self.method_interp = method_interp
 
        # Tell solve_ivp that it should use the Stop_Function to terminate the 
        # integration.  Our stop functions made sure that we stay within the 
        # problem's domain.  See definitions of trace_stop_... in trace_stop_funcs.py.
        self.Stop_Function.terminal = True
        
        # Make grid for s from 0 to max_length, solve_ivp starts at 
        # s_grid[0] and runs until the end (unless aborted by Stop_Function).
        # s is the distance down the field line.
        self.s_grid = np.arange(0,max_length+grid_spacing,grid_spacing)
        
        return
    
    def trace_field_line(self, X0, forward):
        """Trace a field line based on start point XO and the provided field.  
        The field is defined by an regular grid of points (x,y,z) at which the 
        field (Fx,Fy,Fz) is provided.  The algorithm uses solve_ivp to step 
        along the field line
        
        Inputs:
            X0 = point at which to start the field line integration
            forward = proceed forward along field line (+step) or backwards (-step)
            
        Outputs:
            field_line.y = x,y,z points along field line starting at X0
        """
       
        logging.info('Tracing field line...' + str(X0) + ' ' + str(forward))
                
        # Function dXds is used by solve_ivp to solve ODE,
        # dX/ds = dF/|F|, which is used for tracing field lines.
        # X (position) and F (field) are vectors.
        # s is the distance down the field line from initial point X0
        def dXds( s,X, xmin, ymin, zmin, xmax, ymax, zmax ):
            F = np.zeros(3)
            F[0] = self.Fx_interpolate(X)
            F[1] = self.Fy_interpolate(X)
            F[2] = self.Fz_interpolate(X)
            if( not forward ):
                F = np.negative( F )
            F_mag = np.linalg.norm(F)
            return F/F_mag
        
        # Call solve_ivp to find field line
        field_line = solve_ivp(fun=dXds, t_span=[self.s_grid[0], self.s_grid[-1]], 
                        y0=X0, t_eval=self.s_grid, rtol=self.tol, 
                        events=self.Stop_Function, method=self.method_ode,
                        args = self.bounds)
        
        # Check for error
        assert( field_line.status != -1 )
        
        return field_line.y
    
    def trace_field_value(self, X):
        """Use the interpolator to estimate the field at point X.
         
        Inputs:
            X = point at which to evaluate field
             
        Outputs:
            F[] = x,y,z components of field at point X
        """

        F = np.zeros(3)
        F[0] = self.Fx_interpolate(X)
        F[1] = self.Fy_interpolate(X)
        F[2] = self.Fz_interpolate(X)
        
        return F
        
class multitrace_cartesian_function(multitrace_cartesian_grid_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through a function that calculates the field (Fx,Fy,Fz) at point 
    (x,y,z).  Algorithm uses solve_ivp to step along the field line.
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
            Field_Function = function that determines F field vector at a specified X (x,y,z)
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
        # problem's domain.  See definitions of trace_stop_... in trace_stop_funcs.py.
        self.Stop_Function.terminal = True
        
        # Define box that bounds the domain of the solution.
        # Xmin and Xmax are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace_stop_funcs.py. Some
        # of the trace_stop_... functions ignore these bounds.)
        self.bounds = Xmin + Xmax
        
        # Make grid for s from 0 to max_length, solve_ivp starts at 
        # s_grid[0] and runs until the end (unless aborted by Stop_Function).
        # s is the distance down the field line.
        self.s_grid = np.arange(0,max_length+grid_spacing,grid_spacing)
    
        return
    
    def trace_field_line(self, X0, forward):
        """Trace a field line based on start point XO and the provided field. 
        The algorithm uses solve_ivp to step along the field line
        
        Inputs:
            X0 = point at which to start the field line integration
            forward = proceed forward along field line (+step) or backwards (-step)
            
        Outputs:
            field_line.y = x,y,z points along field line starting at X0
        """
       
        logging.info('Tracing field line...' + str(X0) + ' ' + str(forward))
    
        # Function dXds is used by solve_ivp to solve the ODE,
        # dX/ds = dF/|F|, which is used for tracing field lines.
        # X (position) and F (field) are vectors.
        # s is the distance down the field line from initial point X0
        def dXds( s,X, xmin, ymin, zmin, xmax, ymax, zmax ):
            F = self.Field_Function( X )
            if( not forward ):
                F = np.negative( F )
            F_mag = np.linalg.norm(F)
            return F/F_mag
        
        # Call solve_ivp to find magnetic field line
        field_line = solve_ivp(fun=dXds, t_span=[self.s_grid[0], self.s_grid[-1]], 
                        y0=X0, t_eval=self.s_grid, rtol=self.tol, 
                        events=self.Stop_Function, method=self.method_ode,
                        args = self.bounds)
        
        # Check for error
        assert( field_line.status != -1 )
        
        return field_line.y

    def trace_field_value(self, X):
        """Use the field function to calculate the field at point X.
         
        Inputs:
            X = point at which to evaluate field 
             
        Outputs:
            F[] = x,y,z components of field at point X
        """

        F = self.Field_Function( X )
        
        return F

class multitrace_cartesian_regular_grid(multitrace_cartesian_grid_base):
    """Trace multiple field lines through the provided field. The field 
    is defined through a regular grid (array) of points (x,y,z) at which the 
    field (Fx,Fy,Fz) is provided.  The locations of the x,y,z points are regularly 
    spaced.  Algorithm uses solve_ivp to step along the field line.
    """
    
    def __init__(self, x, y, z, Fx, Fy, Fz,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest' ):
        """Initialize multitrace_cartesian_regular_grid()
        
        Inputs:
            x, y, z = x,y,z each is a range of values defining the positions at 
                which the field is known.  x, y, and z vectors define
                extent of grid along each axis.
            Fx, Fy, Fz = value of field vector at each x,y,z
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            
        Outputs:
            None
        """
        super().__init__(Stop_Function, 
                        tol, 
                        grid_spacing, 
                        max_length, 
                        method_ode,
                        method_interp )
        
        logging.info('Initializing multitrace for regular grid: ' + str(tol) + 
                     ' ' + str(grid_spacing) + ' ' + str(max_length) + ' ' + 
                     method_ode + ' ' + method_interp)

        # Input error checks
        # assert isinstance(x, types.ndarray)
        # assert isinstance(y, types.ndarray)
        # assert isinstance(z, types.ndarray)
        # assert isinstance(Fx, types.ndarray)
        # assert isinstance(Fy, types.ndarray)
        # assert isinstance(Fz, types.ndarray)
        
        # Store instance data
        self.x = x
        self.y = y
        self.z = z
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz
 
        # Create interpolators for regular grid
        self.Fx_interpolate = RegularGridInterpolator((x, y, z), Fx, method=method_interp)
        self.Fy_interpolate = RegularGridInterpolator((x, y, z), Fy, method=method_interp)
        self.Fz_interpolate = RegularGridInterpolator((x, y, z), Fz, method=method_interp)
        
        # Define box that bounds the domain of the solution.
        # min and max are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace_stop_funcs.py. Some
        # of the trace_stop_... functions ignore these bounds.)
        self.bounds = [min(x), min(y), min(z)] + [max(x), max(y), max(z)]
        
        return
    
class multitrace_cartesian_unstructured(multitrace_cartesian_grid_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured data, an array of points (x,y,z) at which the 
    field (Fx,Fy,Fz) is provided.  The locations of the x,y,z points are 
    unconstrained.  Algorithm uses solve_ivp to step along the field line
    """

    def __init__(self, x, y, z, Fx, Fy, Fz,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest' ):
 
        """Initialize multitrace_cartesian_unstructured
            
        Inputs:
            x, y, z = define the positions at which the field is known 
            Fx, Fy, Fz = value of field vector at each x,y,z
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            
        Outputs:
            None
        """
        super().__init__(Stop_Function, 
                        tol, 
                        grid_spacing, 
                        max_length, 
                        method_ode,
                        method_interp )
        
        logging.info('Initializing multitrace for unstructured data: ' + str(tol) + 
                     ' ' + str(grid_spacing) + ' ' + str(max_length) + ' ' + 
                     method_ode + ' ' + method_interp)

        # Input error checks
        # assert isinstance(x, types.ndarray)
        # assert isinstance(y, types.ndarray)
        # assert isinstance(z, types.ndarray)
        # assert isinstance(Fx, types.ndarray)
        # assert isinstance(Fy, types.ndarray)
        # assert isinstance(Fz, types.ndarray)

        # Store instance data
        self.x = x
        self.y = y
        self.z = z
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz
        
        # Create interpolators for unstructured data
        # Use list(zip(...)) to convert x,y,z's => (x0,y0,z0), (x1,y1,z1), ...
        if( method_interp == 'linear'):
            self.Fx_interpolate = LinearNDInterpolator(list(zip(x, y, z)), Fx )
            self.Fy_interpolate = LinearNDInterpolator(list(zip(x, y, z)), Fy )
            self.Fz_interpolate = LinearNDInterpolator(list(zip(x, y, z)), Fz )
        if( method_interp == 'nearest'):
            self.Fx_interpolate = NearestNDInterpolator(list(zip(x, y, z)), Fx )
            self.Fy_interpolate = NearestNDInterpolator(list(zip(x, y, z)), Fy )
            self.Fz_interpolate = NearestNDInterpolator(list(zip(x, y, z)), Fz )
 
       # Define box that bounds the domain of the solution.
        # min and max are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace_stop_funcs.py. Some
        # of the trace_stop_... functions ignore these bounds.)
        self.bounds = [min(x), min(y), min(z)] + [max(x), max(y), max(z)]
        
        return

class multitrace_cartesian_unstructured_swmfio(multitrace_cartesian_grid_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured data, an array of points (x,y,z) at which the 
    field (Fx,Fy,Fz) is provided.  The locations of the x,y,z points are 
    unconstrained.  Algorithm uses solve_ivp to step along the field line, and
    the swmfio BATSRUS interpolator
    """
    import swmfio
    
    def __init__(self, x, y, z, Fx, Fy, Fz,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    batsrus = None,
                    Fx_field = 'bx', Fy_field = 'by', Fz_field = 'bz'):
 
        """Initialize multitrace_cartesian_unstructured
            
        Inputs:
            x, y, z = define the positions at which the field is known 
            Fx, Fy, Fz = value of field vector at each x,y,z
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            
        Outputs:
            None
        """
        super().__init__(Stop_Function, 
                        tol, 
                        grid_spacing, 
                        max_length, 
                        method_ode )
        
        logging.info('Initializing multitrace for unstructured data and swmfio interpolation: ' 
                     + str(tol) + ' ' + str(grid_spacing) + ' ' + 
                     str(max_length) + ' ' + method_ode)

        # Input error checks
        # assert isinstance(x, types.ndarray)
        # assert isinstance(y, types.ndarray)
        # assert isinstance(z, types.ndarray)
        # assert isinstance(Fx, types.ndarray)
        # assert isinstance(Fy, types.ndarray)
        # assert isinstance(Fz, types.ndarray)
        assert( batsrus != None )
        assert isinstance( Fx_field, str  )
        assert isinstance( Fy_field, str  )
        assert isinstance( Fz_field, str  )

        # Store instance data
        self.x = x
        self.y = y
        self.z = z
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz
        self.batsrus = batsrus
        self.Fx_field = Fx_field
        self.Fy_field = Fy_field
        self.Fz_field = Fz_field        
        
        # Define box that bounds the domain of the solution.
        # min and max are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace_stop_funcs.py. Some
        # of the trace_stop_... functions ignore these bounds.)
        self.bounds = [min(x), min(y), min(z)] + [max(x), max(y), max(z)]
        
        # Make grid for s from 0 to max_length, solve_ivp starts at 
        # s_grid[0] and runs until the end (unless aborted by Stop_Function).
        # s is the distance down the field line.
        self.s_grid = np.arange(0,max_length+grid_spacing,grid_spacing)

        return

    def trace_field_line(self, X0, forward):
        """Trace a field line based on start point XO and the provided field.  
        The field is defined by an array of points (x,y,z) at which the field 
        (Fx,Fy,Fz) is provided.  The algorithm uses solve_ivp to step along the 
        field line
        
        Inputs:
            X0 = point at which to start the field line integration
            forward = proceed forward along field line (+step) or backwards (-step)
            
        Outputs:
            field_line.y = x,y,z points along field line starting at X0
        """
       
        logging.info('Tracing field line...' + str(X0) + ' ' + str(forward))
        
        # Function dXds is used by solve_ivp to solve ODE,
        # dX/ds = dF/|F|, which is used for tracing field lines.
        # X (position) and F (field) are vectors.
        # s is the distance down the field line from initial point X0
        def dXds( s,X, xmin, ymin, zmin, xmax, ymax, zmax ):
            F = np.zeros(3)
            F[0] = self.batsrus.interpolate( X, self.Fx_field )
            F[1] = self.batsrus.interpolate( X, self.Fy_field )
            F[2] = self.batsrus.interpolate( X, self.Fz_field )
            if( not forward ):
                F = np.negative( F )
            F_mag = np.linalg.norm(F)
            return F/F_mag
        
        # Call solve_ivp to find field line
        field_line = solve_ivp(fun=dXds, t_span=[self.s_grid[0], self.s_grid[-1]], 
                        y0=X0, t_eval=self.s_grid, rtol=self.tol, 
                        events=self.Stop_Function, method=self.method_ode,
                        args = self.bounds)
        
        # Check for error
        assert( field_line.status != -1 )
        
        return field_line.y
        
    def trace_field_value(self, X):
        """Use the interpolator to estimate the field at point X.
         
        Inputs:
            X = point at which to evaluate field
             
        Outputs:
            F[] = x,y,z components of field at point X
        """

        F = np.zeros(3)
        F[0] = self.batsrus.interpolate( X, self.Fx_field )
        F[1] = self.batsrus.interpolate( X, self.Fy_field )
        F[2] = self.batsrus.interpolate( X, self.Fz_field )
    
        return F
    