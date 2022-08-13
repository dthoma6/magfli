#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:56:41 2022

@author: Dean Thomas
"""

from magfli.multitrace import multitrace_cartesian_function, \
        multitrace_cartesian_regular_grid, \
        multitrace_cartesian_unstructured, \
        multitrace_cartesian_unstructured_swmfio
from magfli.trace_stop_funcs import trace_stop_earth
from magfli.dipole import dipole_earth_cartesian
import types
import logging

from enum import Enum, auto
class integrate_direction(Enum):
    forward = auto()
    backward = auto()
    both = auto()

# # https://docs.python.org/3/library/multiprocessing.html#using-a-pool-of-workers
# def solve(self, num_startpts, n):
#     logging.info("Working on start point {}".format(n))
    
#     # When both directions selected, put forward line at n and backward
#     # line at n+num_pts
    
#     if( self.direction == integrate_direction.both ):
#         self.fieldlines[n] = self.multitrace.trace_field_line(self.start_pts[n], True)
#         self.fieldlines[n+num_startpts] = self.multitrace.trace_field_line(self.start_pts[n], False)
#     else:      
#         self.fieldlines[n] = self.multitrace.trace_field_line(self.start_pts[n], 
#                                 self.direction == integrate_direction.forward)

class fieldlines_base():
    """Base class for fieldline classes.  It includes all of the procedures
    common across the fieldlines classes.  Most of the changes should be to
    __init__().  set_start_points() and trace_field_lines() will be unchanged
    in most situations.
    """
    def __init__(self, Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    start_pts = None,
                    direction = integrate_direction.forward):
        

        """Initialize fieldlines_base class, which includes things common to
        child classes.
            
        Inputs:
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp 
                for details
            start_pts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
        Outputs:
            None
        """
        logging.info('Initializing fieldlines for fieldlines base class: ' 
                      + str(tol) + ' ' + str(grid_spacing) + ' ' + str(max_length) 
                      + ' ' + method_ode)

        # Check inputs
        if start_pts != None:
            assert isinstance(start_pts, list)
        if direction != None:
            assert isinstance(direction, integrate_direction)
        assert isinstance(Stop_Function, types.FunctionType)
        assert( method_ode == 'RK23' or method_ode == 'RK45' or method_ode == 'DOP853')

        # Store inputs
        
        # Initialize the start points for the field line tracing, along
        # with whether we integrate forward, backwards, or both
        self.start_pts = start_pts
        self.direction = direction

        # Initialize storage for field lines. One field line for each
        # start point, unless we go in both directions and there are two
        # for each start point
        if( start_pts != None ):
            if (direction == integrate_direction.forward or
                direction == integrate_direction.backward):
                self.fieldlines = [None]*len(start_pts)
            else:
                self.fieldlines = [None]*2*len(start_pts)
                
        return
    
    def set_start_points(self, start_pts=None, 
                        direction=integrate_direction.forward):
        """Set start points and direction of integration
            
        Inputs:
            start_pts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            
        Outputs:
            None
        """

        logging.info('Setting start points field line...')

        assert isinstance(start_pts, list)
        assert len(start_pts) >= 1
        assert isinstance(direction, integrate_direction)
 
        self.start_pts = start_pts
        self.direction = direction

        # Initialize storage for field lines. One field line for each
        # start point, unless we go in both directions and there are two
        # for each start point
        if (direction == integrate_direction.forward or
            direction == integrate_direction.backward):
            self.fieldlines = [None]*len(start_pts)
        else:
            self.fieldlines = [None]*2*len(start_pts)

        return
    
    def trace_field_lines(self):
        """Trace field lines from each start point.  Trace will go forward,
        backward, or both directions depending on the internal value of 
        direction.  Direction set during class initialization or by call to
        setstartpoints.
            
        Inputs:
            None
            
        Outputs:
            None
        """
        logging.info('Tracing field lines...')

        num_startpts = len(self.start_pts)
        
        for i in range(num_startpts):
            # Trace field line
            if(self.direction != integrate_direction.both):
                self.fieldlines[i] = self.multitrace.trace_field_line(self.start_pts[i], 
                                        self.direction == integrate_direction.forward)
            else:
                self.fieldlines[i] = self.multitrace.trace_field_line(self.start_pts[i], True)
                self.fieldlines[i+num_startpts] = self.multitrace.trace_field_line(self.start_pts[i], False)
 
        # parallel = True
        # if parallel:
        #     import multiprocessing as mp
        #     num_cores = mp.cpu_count()
        #     num_cores = min(num_cores, num_startpts, 20)
        #     logging.info(f'Parallel processing {num_startpts} field lines using {num_cores} cores')   
        #     p = mp.Pool(num_cores)
        #     n = range(num_startpts)
        #     p.map(solve, self, num_startpts, n)
        # else:
        #     logging.info(f'Serial processing {num_startpts} field lines')
        #     for n in range(num_startpts):
        #         solve(self, num_startpts, n)

        return self.fieldlines    
    
    def trace_field_value(self, X):
        """Use the interpolator to estimate the field at point X.
         
        Inputs:
            X = point at which to evaluate field
             
        Outputs:
            F[] = x,y,z components of field at point X
        """
        return self.multitrace.trace_field_value(X)

class fieldlines_cartesian_function(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through function that determines the field (Fx,Fy,Fz) at at given
    (x,y,z) point. Algorithm uses solve_ivp to step along the 
    field line.  
    """

    def __init__(self, Xmin, Xmax, 
                    Field_Function = dipole_earth_cartesian,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    start_pts = None,
                    direction = integrate_direction.forward):
        
        """Initialize fieldlines_cartesian_unstructured_BATSRUSfile
            
        Inputs:
            x, y, z = define the positions at which the field is known 
            Fx, Fy, Fz = value of field vector at each x,y,z
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            start_pts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            x_field,y_field,z_field = string defining which arrays in the 
                BATSRUS file have the x,y,z grid points
            Fx_field,Fy_field,Fz_field = strings defining which arrays in the 
                BATSRUS file contains data on the field Fx, Fy, and Fz 
        Outputs:
            None
        """
        super().__init__(Stop_Function, 
                        tol, 
                        grid_spacing, 
                        max_length, 
                        method_ode,
                        start_pts,
                        direction)
        
        logging.info('Initializing fieldlines for function: ' 
                      + str(tol) + ' ' + str(grid_spacing) + ' ' + str(max_length) 
                      + ' ' + method_ode )

        # Verify other inputs
        # assert isinstance(x, types.ndarray)
        # assert isinstance(y, types.ndarray)
        # assert isinstance(z, types.ndarray)
        # assert isinstance(Fx, types.ndarray)
        # assert isinstance(Fy, types.ndarray)
        # assert isinstance(Fz, types.ndarray)
        assert isinstance(Field_Function, types.FunctionType)
                       
        # Initialize multitrace for tracing field lines
        self.multitrace = multitrace_cartesian_function( Xmin, Xmax,
                                        Field_Function = Field_Function,
                                        Stop_Function = Stop_Function, 
                                        tol = tol, grid_spacing = grid_spacing, 
                                        max_length = max_length, 
                                        method_ode = method_ode )
        
        return

class fieldlines_cartesian_regular_grid(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through regular grid.  The grid is an array of points (x,y,z) 
    at which the field (Fx,Fy,Fz) is provided.  The locations of the x,y,z 
    points are regularly spaced.  Algorithm uses solve_ivp to step along the 
    field line.  
    """

    def __init__(self, x, y, z, Fx, Fy, Fz,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest',
                    start_pts = None,
                    direction = integrate_direction.forward):
        
        """Initialize fieldlines_cartesian_unstructured_BATSRUSfile
            
        Inputs:
            x, y, z = define the positions at which the field is known 
            Fx, Fy, Fz = value of field vector at each x,y,z
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            start_pts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            x_field,y_field,z_field = string defining which arrays in the 
                BATSRUS file have the x,y,z grid points
            Fx_field,Fy_field,Fz_field = strings defining which arrays in the 
                BATSRUS file contains data on the field Fx, Fy, and Fz 
        Outputs:
            None
        """
        super().__init__(Stop_Function, 
                        tol, 
                        grid_spacing, 
                        max_length, 
                        method_ode,
                        start_pts,
                        direction)
        
        logging.info('Initializing fieldlines for regular grid: ' 
                      + str(tol) + ' ' + str(grid_spacing) + ' ' + str(max_length) 
                      + ' ' + method_ode + ' ' + method_interp )

        # Verify other inputs
        # assert isinstance(x, types.ndarray)
        # assert isinstance(y, types.ndarray)
        # assert isinstance(z, types.ndarray)
        # assert isinstance(Fx, types.ndarray)
        # assert isinstance(Fy, types.ndarray)
        # assert isinstance(Fz, types.ndarray)
        assert( method_interp == 'linear' or method_interp == 'nearest')
                       
        # Initialize multitrace for tracing field lines
        self.multitrace = multitrace_cartesian_regular_grid( x, y, z, Fx, Fy, Fz,
                                        Stop_Function = Stop_Function, 
                                        tol = tol, grid_spacing = grid_spacing, 
                                        max_length = max_length, 
                                        method_ode = method_ode, 
                                        method_interp = method_interp )
        
        return

class fieldlines_cartesian_unstructured(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured grid.  The grid is an array of points (x,y,z) 
    at which the field (Fx,Fy,Fz) is provided.  The locations of the x,y,z 
    points is unconstrained.  Algorithm uses solve_ivp to step along the field line.  
    """

    def __init__(self, x, y, z, Fx, Fy, Fz,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest',
                    start_pts = None,
                    direction = integrate_direction.forward):
        
        """Initialize fieldlines_cartesian_unstructured_BATSRUSfile
            
        Inputs:
            x, y, z = define the positions at which the field is known 
            Fx, Fy, Fz = value of field vector at each x,y,z
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            start_pts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            x_field,y_field,z_field = string defining which arrays in the 
                BATSRUS file have the x,y,z grid points
            Fx_field,Fy_field,Fz_field = strings defining which arrays in the 
                BATSRUS file contains data on the field Fx, Fy, and Fz 
        Outputs:
            None
        """
        super().__init__(Stop_Function, 
                        tol, 
                        grid_spacing, 
                        max_length, 
                        method_ode,
                        start_pts,
                        direction)
        
        logging.info('Initializing fieldlines for unstructured grid: ' 
                      + str(tol) + ' ' + str(grid_spacing) + ' ' + str(max_length) 
                      + ' ' + method_ode + ' ' + method_interp )

        # Verify other inputs
        # assert isinstance(x, types.ndarray)
        # assert isinstance(y, types.ndarray)
        # assert isinstance(z, types.ndarray)
        # assert isinstance(Fx, types.ndarray)
        # assert isinstance(Fy, types.ndarray)
        # assert isinstance(Fz, types.ndarray)
        assert( method_interp == 'linear' or method_interp == 'nearest')
                       
        # Initialize multitrace for tracing field lines
        self.multitrace = multitrace_cartesian_unstructured( x, y, z, Fx, Fy, Fz,
                                        Stop_Function = Stop_Function, 
                                        tol = tol, grid_spacing = grid_spacing, 
                                        max_length = max_length, 
                                        method_ode = method_ode, 
                                        method_interp = method_interp )
        
        return

class fieldlines_cartesian_unstructured_BATSRUSfile(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured grid stored in a BATSRUS file.  The grid is an
    array of points (x,y,z) at which the field (Fx,Fy,Fz) is provided.  
    The locations of the x,y,z points is unconstrained.  Algorithm uses 
    solve_ivp to step along the field line.  
    """

    def __init__(self, filename = None,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest',
                    start_pts = None,
                    direction = integrate_direction.forward,
                    x_field = 'x', y_field = 'y', z_field = 'z',
                    Fx_field = 'bx', Fy_field = 'by', Fz_field = 'bz'):
        
        """Initialize fieldlines_cartesian_unstructured_BATSRUSfile
            
        Inputs:
            filename = path for BATSRUS file with unstructured cartesian grid, 
                which includes x,y,z grid points and field F at each grid point
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            start_pts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            x_field,y_field,z_field = string defining which arrays in the 
                BATSRUS file have the x,y,z grid points
            Fx_field,Fy_field,Fz_field = strings defining which arrays in the 
                BATSRUS file contains data on the field Fx, Fy, and Fz 
        Outputs:
            None
        """
        super().__init__(Stop_Function, 
                        tol, 
                        grid_spacing, 
                        max_length, 
                        method_ode,
                        start_pts,
                        direction)
        
        logging.info('Initializing fieldlines for unstructured BATSRUS grid: ' 
                      + str(tol) + ' ' + str(grid_spacing) + ' ' + str(max_length) 
                      + ' ' + method_ode + ' ' + method_interp + ' '
                      + filename + '.{tree, info, out}')

        # Store input variable
        self.filename = filename

        # Verify BATSRUS file exists
        from os.path import exists
        assert exists(filename + '.out')
        assert exists(filename + '.info')
        assert exists(filename + '.tree')
        
        # Verify other inputs
        assert( method_interp == 'linear' or method_interp == 'nearest')
        assert isinstance( x_field, str )
        assert isinstance( y_field, str )
        assert isinstance( z_field, str )
        assert isinstance( Fx_field, str )
        assert isinstance( Fy_field, str )
        assert isinstance( Fz_field, str )

        import swmfio
        batsclass = swmfio.read_batsrus(filename)
           
        # Extract x, y, z and Fx, Fy, Fz from file
        var_dict = dict(batsclass.varidx)
        x = batsclass.data_arr[:,var_dict[x_field]][:]
        y = batsclass.data_arr[:,var_dict[y_field]][:]
        z = batsclass.data_arr[:,var_dict[z_field]][:]
            
        Fx = batsclass.data_arr[:,var_dict[Fx_field]][:]
        Fy = batsclass.data_arr[:,var_dict[Fy_field]][:]
        Fz = batsclass.data_arr[:,var_dict[Fz_field]][:]
            
        # Initialize multitrace for tracing field lines
        self.multitrace = multitrace_cartesian_unstructured( x, y, z, Fx, Fy, Fz,
                                        Stop_Function = Stop_Function, 
                                        tol = tol, grid_spacing = grid_spacing, 
                                        max_length = max_length, 
                                        method_ode = method_ode, 
                                        method_interp = method_interp )
        
        return
 
class fieldlines_cartesian_unstructured_swmfio_BATSRUSfile(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured grid stored in a BATSRUS file.  The grid is an
    array of points (x,y,z) at which the field (Fx,Fy,Fz) is provided.  
    The locations of the x,y,z points is unconstrained.  Algorithm uses 
    solve_ivp to step along the field line.  
    """

    def __init__(self, filename = None,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'swmfio',
                    start_pts = None,
                    direction = integrate_direction.forward,
                    x_field = 'x', y_field = 'y', z_field = 'z',
                    Fx_field = 'bx', Fy_field = 'by', Fz_field = 'bz'):
        
        """Initialize fieldlines_cartesian_unstructured_swmfio_BATSRUSfile
            
        Inputs:
            filename = path for BATSRUS file with unstructured cartesian grid, 
                which includes x,y,z grid points and field F at center of cells
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            start_pts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            x_field,y_field,z_field = string defining which arrays in the 
                BATSRUS file have the x,y,z grid points
            Fx_field,Fy_field,Fz_field = strings defining which arrays in the 
                BATSRUS file contains data on the field Fx, Fy, and Fz 
        Outputs:
            None
        """
        super().__init__(Stop_Function, 
                        tol, 
                        grid_spacing, 
                        max_length, 
                        method_ode,
                        start_pts,
                        direction)
        
        logging.info('Initializing fieldlines for unstructured BATSRUS grid: ' 
                      + str(tol) + ' ' + str(grid_spacing) + ' ' + str(max_length) 
                      + ' ' + method_ode + ' ' + method_interp + ' '
                      + filename + '.{tree, info, out}')

        # Store input variable
        self.filename = filename

        # Verify BATSRUS file exists
        from os.path import exists
        assert exists(filename + '.out')
        assert exists(filename + '.info')
        assert exists(filename + '.tree')
        
        # Verify other inputs
        assert( method_interp == 'swmfio')
        assert isinstance( x_field, str )
        assert isinstance( y_field, str )
        assert isinstance( z_field, str )
        assert isinstance( Fx_field, str )
        assert isinstance( Fy_field, str )
        assert isinstance( Fz_field, str )

        import swmfio
        batsclass = swmfio.read_batsrus(filename)
           
        # Extract x, y, z and Fx, Fy, Fz from file
        var_dict = dict(batsclass.varidx)
        x = batsclass.data_arr[:,var_dict[x_field]][:]
        y = batsclass.data_arr[:,var_dict[y_field]][:]
        z = batsclass.data_arr[:,var_dict[z_field]][:]
            
        Fx = batsclass.data_arr[:,var_dict[Fx_field]][:]
        Fy = batsclass.data_arr[:,var_dict[Fy_field]][:]
        Fz = batsclass.data_arr[:,var_dict[Fz_field]][:]

        # Initialize multitrace for tracing field lines
        self.multitrace = multitrace_cartesian_unstructured_swmfio( x, y, z, Fx, Fy, Fz,
                                        Stop_Function = Stop_Function, 
                                        tol = tol, grid_spacing = grid_spacing, 
                                        max_length = max_length, 
                                        method_ode = method_ode, 
                                        batsrus = batsclass,
                                        Fx_field = Fx_field, 
                                        Fy_field = Fy_field, 
                                        Fz_field = Fz_field )
        
        return

class fieldlines_cartesian_unstructured_VTKfile(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured grid stored in a VTK file.  The grid is an
    array of points (x,y,z) at which the field (Fx,Fy,Fz) is provided.  
    The locations of the x,y,z points is unconstrained.  Algorithm uses 
    solve_ivp to step along the field line.  
    """

    def __init__(self, filename = None,
                    Stop_Function = trace_stop_earth, 
                    tol = 1e-5, 
                    grid_spacing = 0.01, 
                    max_length = 100, 
                    method_ode = 'RK23',
                    method_interp = 'nearest',
                    start_pts = None,
                    direction = integrate_direction.forward,
                    F_field = 'b',
                    cell_centers = None):
        
        """Initialize fieldlines_cartesian_unstructured_VTKfile
            
        Inputs:
            filename = path for VTK file with unstructured cartesian grid, which
                includes x,y,z grid points and field F at center of cells
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            start_pts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            field = a string defining which cell array in the VTK file contains
                data on the field F
        Outputs:
            None
        """
        super().__init__(Stop_Function, 
                        tol, 
                        grid_spacing, 
                        max_length, 
                        method_ode,
                        start_pts,
                        direction)
        
        logging.info('Initializing fieldlines for unstructured VTK grid: ' + str(tol) + 
                      ' ' + str(grid_spacing) + ' ' + str(max_length) + ' ' + 
                      method_ode + ' ' + method_interp + ' ' + filename)

        # Store input variable
        self.filename = filename

        # Verify VTK file exists
        from os.path import exists
        assert exists(filename)
        
        # Check other inputs
        assert( method_interp == 'linear' or method_interp == 'nearest')
        assert isinstance( F_field, str )
        assert (isinstance( cell_centers, str) or (cell_centers is None))
        
        from vtk import vtkUnstructuredGridReader
        from paraview.vtk import vtkCellCenters
        from paraview.vtk.util import numpy_support as vn
        
        # Open VTK file which should contain an unstructured cartesian grid
        reader = vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()

        # Extract data from VTK file
        data = reader.GetOutput()

        # The x,y,z points in the unstructured grid are offset from
        # the center of the cells.  We don't use pts
        #pts = vn.vtk_to_numpy(data.GetPoints().GetData())

        # So we read the field F from the VTK file and use 
        # vtkCellCenters to find the F x,y,z positions
        
        # The field F at each cell center
        F = vn.vtk_to_numpy(data.GetCellData().GetArray(F_field))
        Fx = F[:,0]
        Fy = F[:,1]
        Fz = F[:,2]
        
        # We need the x,y,z locations of the cell centers.
        # If Cell_centers is str, we read them from the file.
        # If Cell_centers is None, we calculate them via VTK.  
        if( isinstance( cell_centers, str ) ):
            C = vn.vtk_to_numpy(data.GetCellData().GetArray(cell_centers))
            x = C[:,0]
            y = C[:,1]
            z = C[:,2]
        else:
            cellCenters = vtkCellCenters()
            cellCenters.SetInputDataObject(data)
            cellCenters.Update()
            Cpts = vn.vtk_to_numpy(cellCenters.GetOutput().GetPoints().GetData())
            x = Cpts[:,0]
            y = Cpts[:,1]
            z = Cpts[:,2]
        
        # Initialize multitrace for tracing field lines
        self.multitrace = multitrace_cartesian_unstructured( x, y, z, Fx, Fy, Fz,
                                        Stop_Function = Stop_Function, 
                                        tol = tol, grid_spacing = grid_spacing, 
                                        max_length = max_length, 
                                        method_ode = method_ode, 
                                        method_interp = method_interp )
        
        return
    
    
class fieldlines_cartesian_unstructured_paraview_VTKfile():
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured grid stored in a VTK file.  The grid is an
    array of points (x,y,z) at which the field (Fx,Fy,Fz) is provided.  
    The locations of the x,y,z points is unconstrained.  Algorithm uses 
    Paraview's StreamTracer to step along the field line.  
    """
    
    def __init__( self, filename = None,
                    max_length = 100, 
                    method_ode = 'RK2',
                    seed_type = 'Line',
                    start_pt1 = None,
                    start_pt2 = None,
                    radius = None,
                    resolution = None,
                    direction = integrate_direction.both,
                    F_field = 'b'):
        """Initialize fieldlines_cartesian_unstructured_paraview_VTKfile
            
        Inputs:
            filename = path for VTK file with unstructured cartesian grid, which
                includes x,y,z grid points and field F at center of cells
            max_length = maximum length (s) of field line
            method_ode = ODE algorithm, RK2, RK4, and RK45, see StreamTracer for details
            seed_type, start_pt1, start_pt2, and resolution = define list of start points 
                see StreamTracer line source for details
            seed_type, start_pt1, radius, and resolution = define list of start points 
                see StreamTracer point cloud source for details
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            field = a string defining which cell array in the VTK file contains
                data on the field F
        Outputs:
            None
        """
        logging.info('Initializing fieldlines for paraview VTK file: ')
    
        # Verify VTK file exists
        from os.path import exists
        assert exists(filename)
    
        # Check inputs
        
        assert( seed_type == 'Line' or seed_type == 'Point Cloud')
        
        if start_pt1 != None and start_pt2 != None:
            assert isinstance(start_pt1, list)
            assert isinstance(start_pt2, list)
        if resolution != None:
            assert isinstance(resolution, int)
        if direction != None:
            assert isinstance(direction, integrate_direction)
    
        assert( method_ode == 'RK2' or method_ode == 'RK4' or method_ode == 'RK45')
        
        # Save inputs
        self.max_length = max_length
        self.method_ode = method_ode
        self.seed_type = seed_type
        self.start_pt1 = start_pt1
        self.start_pt2 = start_pt2
        self.radius = radius
        self.resolution = resolution
        self.direction = direction
        self.F_field = F_field
    
        # Initialize storage for field lines. One field line for each
        # start point, unless we go in both directions and there are two
        # for each start point
        if( seed_type == 'Line' ):
            if( start_pt1 != None and start_pt2 != None and resolution != None ):
                if (direction == integrate_direction.forward or
                    direction == integrate_direction.backward):
                    self.fieldlines = [None]*(resolution+1)
                else:
                    self.fieldlines = [None]*2*(resolution+1)
        if( seed_type == 'Point Cloud' ):
            if( start_pt1 != None and radius != None and resolution != None ):
                if (direction == integrate_direction.forward or
                    direction == integrate_direction.backward):
                    self.fieldlines = [None]*(resolution)
                else:
                    self.fieldlines = [None]*2*(resolution)
                
        import magnetovis as mvs
        import paraview.simple as pvs
     
        # read file
        self.batsrus = mvs.BATSRUS(file=filename)
    
        # find source
        self.registrationName = list(pvs.GetSources().keys())\
            [list(pvs.GetSources().values()).index(pvs.GetActiveSource())][0]
        self.bat_source = pvs.FindSource(self.registrationName)
        return
    
    def trace_field_lines(self):
        """Trace field lines from each start point.  Trace will go forward,
        backward, or both directions depending on the internal value of 
        direction.  Direction set during class initialization.
            
        Inputs:
            None
            
        Outputs:
            None
        """
        logging.info('Tracing field lines...')
       
        import paraview.simple as pvs
        from paraview import servermanager as sm
        from paraview.vtk.util import numpy_support as vn
        from paraview import vtk as vtk
        import matplotlib.pyplot as plt
    
        # Create a new 'Stream Tracer' using the F_field for tracing field lines
        streamTracer = pvs.StreamTracer(registrationName='StreamTracer1', 
            Input=self.bat_source, SeedType='Line')
        streamTracer.Vectors = ['CELLS', self.F_field]
        streamTracer.MaximumStreamlineLength = self.max_length
     
        # Set up seed points
        if( self.seed_type == 'Line' ):
            streamTracer.SeedType = 'Line'
            streamTracer.SeedType.Point1 = self.start_pt1
            streamTracer.SeedType.Point2 = self.start_pt2
            streamTracer.SeedType.Resolution = self.resolution
        if( self.seed_type == 'Point Cloud' ):
            streamTracer.SeedType = 'Point Cloud'
            streamTracer.SeedType.Center = self.start_pt1
            streamTracer.SeedType.Radius = self.radius
            streamTracer.SeedType.NumberOfPoints = self.resolution
        
        # Initialize integration direction
        if (self.direction == integrate_direction.forward):
            streamTracer.IntegrationDirection = 'FORWARD'
        if (self.direction == integrate_direction.backward):
            streamTracer.IntegrationDirection = 'BACKWARD'
        if (self.direction == integrate_direction.both):
            streamTracer.IntegrationDirection = 'BOTH'
         
        # Select ode solver
        if (self.method_ode == 'RK2'):
            streamTracer.IntegratorType = 'Runge-Kutta 2'
        if (self.method_ode == 'RK4'):
            streamTracer.IntegratorType = 'Runge-Kutta 4'
        if (self.method_ode == 'RK45'):
            streamTracer.IntegratorType = 'Runge-Kutta 4-5'
            
        # Note, streamtracer has two interpolators, 'Interpolator with Point Locator'
        # and 'Interpolator with Cell Locator'
        #
        # https://www.paraview.org/Wiki/ParaView/Users_Guide/List_of_filters
        #
        # Point is the default, and cell locator does not work with BATSRUS data
        #streamTracer.InterpolatorType = 'Interpolator with Cell Locator'
        
        # In the commented lines below, I tried changing streamtracer
        # attributes from the default, they had no obvious change to stream lines
        # Obvious means no difference in 3D plot of field lines
        #streamTracer.ComputeVorticity = 0
        #streamTracer.IntegrationStepUnit = 0.01
        #streamTracer.MaximumError = 1e-3
        #streamTracer.MaximumStepLength = 1.
        #streamTracer.MinimumStepLength = 0.001
        #streamTracer.SurfaceStreamlines = 1
        #streamTracer.TerminalSpeed = 1e-20
        
        # In the two commented lines below, these attributes did have an affect
        #streamTracer.InitialStepLength = 0.01
        #streamTracer.MaximumSteps = 20000

        # print('ComputeVorticity ', streamTracer.ComputeVorticity)
        # print('InitialStepLength ', streamTracer.InitialStepLength)
        # print('IntegrationStepUnit ', streamTracer.IntegrationStepUnit)
        # print('MaximumError ', streamTracer.MaximumError)
        # print('MaximumStepLength ', streamTracer.MaximumStepLength)
        # print('MaximumSteps ', streamTracer.MaximumSteps)
        # print('MinimumStepLength ', streamTracer.MinimumStepLength)
        # print('SurfaceStreamlines ', streamTracer.SurfaceStreamlines)
        # print('TerminalSpeed ', streamTracer.TerminalSpeed)

        streamline = sm.Fetch(streamTracer)
        streamlines_points = vn.vtk_to_numpy(streamline.GetPoints().GetData())
        
        vtkLines = streamline.GetLines()
        vtkLines.InitTraversal()
        point_list = vtk.vtkIdList()
        
        i = 0;
        while vtkLines.GetNextCell(point_list):
            start_id = point_list.GetId(0)
            end_id = point_list.GetId(point_list.GetNumberOfIds() - 1)
            # Note: the Paraview lines are the transpose of the scipy lines
            # Hence, we transpose the matrix here so we are consistent
            self.fieldlines[i] = (streamlines_points[start_id:end_id]).T
            i += 1
    
        return self.fieldlines
    
    def reset(self):
        """Reset paraview session.
            
        Inputs:
            None
            
        Outputs:
            None
        """
        import paraview.simple as pvs
        pvs.ResetSession()
