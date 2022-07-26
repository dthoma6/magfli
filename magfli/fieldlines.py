#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:56:41 2022

@author: Dean Thomas
"""

from magfli.multitrace import multitrace_cartesian_unstructured, \
        multitrace_cartesian_unstructured_swmfio
from magfli.trace_stop_funcs import trace_stop_earth
import numpy as np
import types
import logging

from enum import Enum, auto
class integrate_direction(Enum):
    forward = auto()
    backward = auto()
    both = auto()
    
class fieldlines_cartesian_unstructured_BATSRUSfile():
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
                    startPts = None,
                    direction = integrate_direction.forward,
                    x_field = 'x', y_field = 'y', z_field = 'z',
                    Fx_field = 'bx', Fy_field = 'by', Fz_field = 'bz'):
        
        """Initialize fieldlines_cartesian_unstructured_BATSRUSfile
            
        Inputs:
            filename = path for BATSRUS file with unstructured cartesian grid, 
                which includes x,y,z grid points and field F at center of cells
            Stop_Function = decides whether to end solve_ivp early
            tol = tolerence for solve_ivp
            grid_spacing = grid spacing for solve_ivp
            max_length = maximum length (s) of field line
            method_ode = solve_ivp algorithm, e.g., RK23, RK45, see solve_ivp for details
            method_interp = interpolator method, 'linear' or 'nearest'
            startPts = list of start points for starting integration, each
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
        logging.info('Initializing fieldlines for unstructured BATSRUS grid: ' 
                     + str(tol) + ' ' + str(grid_spacing) + ' ' + str(max_length) 
                     + ' ' + method_ode + ' ' + method_interp + ' '
                     + filename + '.{tree, info, out}')

        # Verify BATSRUS file exists
        from os.path import exists
        assert exists(filename + '.out')
        assert exists(filename + '.info')
        assert exists(filename + '.tree')
        
        # Check other inputs
        if startPts != None:
            assert isinstance(startPts, list)
        if direction != None:
            assert isinstance(direction, integrate_direction)
        assert isinstance(Stop_Function, types.FunctionType)
        assert( method_ode == 'RK23' or method_ode == 'RK45' or method_ode == 'DOP853')
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
        x = batsclass.data_arr[:,var_dict['x']][:]
        y = batsclass.data_arr[:,var_dict['y']][:]
        z = batsclass.data_arr[:,var_dict['z']][:]
            
        Fx = batsclass.data_arr[:,var_dict['bx']][:]
        Fy = batsclass.data_arr[:,var_dict['by']][:]
        Fz = batsclass.data_arr[:,var_dict['bz']][:]
            
        # Initialize the start points for the field line tracing, along
        # with whether we integrate forward, backwards, or both
        self.startPts = startPts
        self.direction = direction

        # Initialize storage for field lines. One field line for each
        # start point, unless we go in both directions and there are two
        # for each start point
        if( startPts != None ):
            if (direction == integrate_direction.forward or
                direction == integrate_direction.backward):
                self.fieldlines = [None]*len(startPts)
            else:
                self.fieldlines = [None]*2*len(startPts)
        
        # Initialize multitrace for tracing field lines
        self.multitrace = multitrace_cartesian_unstructured( x, y, z, Fx, Fy, Fz,
                                       Stop_Function = Stop_Function, 
                                       tol = tol, grid_spacing = grid_spacing, 
                                       max_length = max_length, 
                                       method_ode = method_ode, 
                                       method_interp = method_interp )
        
        return
    
    def setstartpoints(self, startPts=None, 
                       direction=integrate_direction.forward):
        """Set start points and direction of integration
            
        Inputs:
            startPts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            
        Outputs:
            None
        """

        logging.info('Setting start points field line...')

        assert isinstance(startPts, list)
        assert len(startPts) >= 1
        assert isinstance(direction, integrate_direction)
 
        self.startPts = startPts
        self.direction = direction

        # Initialize storage for field lines. One field line for each
        # start point, unless we go in both directions and there are two
        # for each start point
        if (direction == integrate_direction.forward or
            direction == integrate_direction.backward):
            self.fieldlines = [None]*len(startPts)
        else:
            self.fieldlines = [None]*2*len(startPts)

        return
    
    def tracefieldlines(self):
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

        num_startpts = len(self.startPts)
        for i in range(num_startpts):
            # Trace field line
            if(self.direction != integrate_direction.both):
                self.fieldlines[i] = self.multitrace.trace_field_line(self.startPts[i], 
                                        self.direction == integrate_direction.forward)
            else:
                self.fieldlines[i] = self.multitrace.trace_field_line(self.startPts[i], True)
                self.fieldlines[i+num_startpts] = self.multitrace.trace_field_line(self.startPts[i], False)

        # def wrap(n):
        #     logging.info("Working on start point {}".format(n))
            
        #     # When both directions selected, put forward line at n and backward
        #     # line at n+num_pts
            
        #     if( self.direction == integrate_direction.both ):
        #         self.fieldlines[n] = self.multitrace.trace_field_line(self.startPts[n], True)
        #         self.fieldlines[n+num_startpts] = self.multitrace.trace_field_line(self.startPts[n], False)
        #     else:      
        #         self.fieldlines[n] = self.multitrace.trace_field_line(self.startPts[n], 
        #                                 self.direction == integrate_direction.forward)

        # if parallel:
        #     from joblib import Parallel, delayed
        #     import multiprocessing
        #     num_cores = multiprocessing.cpu_count()
        #     num_cores = min(num_cores, num_startpts, 20)
        #     logging.info(f'Parallel processing {num_startpts} field lines using {num_cores} cores')            
        #     Parallel(n_jobs=num_cores)(delayed(wrap)(n) for n in range(num_startpts))
        # else:
        #     logging.info(f'Serial processing {num_startpts} field lines')
        #     for n in range(num_startpts):
        #         wrap(n)

        return self.fieldlines    
    
class fieldlines_cartesian_unstructured_swmfio_BATSRUSfile():
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
                    startPts = None,
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
            startPts = list of start points for starting integration, each
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
        logging.info('Initializing fieldlines for unstructured BATSRUS grid: ' 
                     + str(tol) + ' ' + str(grid_spacing) + ' ' + str(max_length) 
                     + ' ' + method_ode + ' ' + method_interp + ' '
                     + filename + '.{tree, info, out}')

        # Verify BATSRUS file exists
        from os.path import exists
        assert exists(filename + '.out')
        assert exists(filename + '.info')
        assert exists(filename + '.tree')
        
        # Check other inputs
        if startPts != None:
            assert isinstance(startPts, list)
        if direction != None:
            assert isinstance(direction, integrate_direction)
        assert isinstance(Stop_Function, types.FunctionType)
        assert( method_ode == 'RK23' or method_ode == 'RK45' or method_ode == 'DOP853')
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
        x = batsclass.data_arr[:,var_dict['x']][:]
        y = batsclass.data_arr[:,var_dict['y']][:]
        z = batsclass.data_arr[:,var_dict['z']][:]
            
        Fx = batsclass.data_arr[:,var_dict['bx']][:]
        Fy = batsclass.data_arr[:,var_dict['by']][:]
        Fz = batsclass.data_arr[:,var_dict['bz']][:]
            
        # Initialize the start points for the field line tracing, along
        # with whether we integrate forward, backwards, or both
        self.startPts = startPts
        self.direction = direction

        # Initialize storage for field lines. One field line for each
        # start point, unless we go in both directions and there are two
        # for each start point
        if( startPts != None ):
            if (direction == integrate_direction.forward or
                direction == integrate_direction.backward):
                self.fieldlines = [None]*len(startPts)
            else:
                self.fieldlines = [None]*2*len(startPts)
        
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
    
    def setstartpoints(self, startPts=None, 
                       direction=integrate_direction.forward):
        """Set start points and direction of integration
            
        Inputs:
            startPts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            
        Outputs:
            None
        """

        logging.info('Setting start points field line...')

        assert isinstance(startPts, list)
        assert len(startPts) >= 1
        assert isinstance(direction, integrate_direction)
 
        self.startPts = startPts
        self.direction = direction

        # Initialize storage for field lines. One field line for each
        # start point, unless we go in both directions and there are two
        # for each start point
        if (direction == integrate_direction.forward or
            direction == integrate_direction.backward):
            self.fieldlines = [None]*len(startPts)
        else:
            self.fieldlines = [None]*2*len(startPts)

        return
    
    def tracefieldlines(self):
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

        num_startpts = len(self.startPts)
        for i in range(num_startpts):
            # Trace field line
            if(self.direction != integrate_direction.both):
                self.fieldlines[i] = self.multitrace.trace_field_line(self.startPts[i], 
                                        self.direction == integrate_direction.forward)
            else:
                self.fieldlines[i] = self.multitrace.trace_field_line(self.startPts[i], True)
                self.fieldlines[i+num_startpts] = self.multitrace.trace_field_line(self.startPts[i], False)

        # def wrap(n):
        #     logging.info("Working on start point {}".format(n))
            
        #     # When both directions selected, put forward line at n and backward
        #     # line at n+num_pts
            
        #     if( self.direction == integrate_direction.both ):
        #         self.fieldlines[n] = self.multitrace.trace_field_line(self.startPts[n], True)
        #         self.fieldlines[n+num_startpts] = self.multitrace.trace_field_line(self.startPts[n], False)
        #     else:      
        #         self.fieldlines[n] = self.multitrace.trace_field_line(self.startPts[n], 
        #                                 self.direction == integrate_direction.forward)

        # if parallel:
        #     from joblib import Parallel, delayed
        #     import multiprocessing
        #     num_cores = multiprocessing.cpu_count()
        #     num_cores = min(num_cores, num_startpts, 20)
        #     logging.info(f'Parallel processing {num_startpts} field lines using {num_cores} cores')            
        #     Parallel(n_jobs=num_cores)(delayed(wrap)(n) for n in range(num_startpts))
        # else:
        #     logging.info(f'Serial processing {num_startpts} field lines')
        #     for n in range(num_startpts):
        #         wrap(n)

        return self.fieldlines    
    
class fieldlines_cartesian_unstructured_VTKfile():
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
                    startPts = None,
                    direction = integrate_direction.forward,
                    F_field = 'b'):
        
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
            startPts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            field = a string defining which cell array in the VTK file contains
                data on the field F
        Outputs:
            None
        """
        logging.info('Initializing fieldlines for unstructured VTK grid: ' + str(tol) + 
                     ' ' + str(grid_spacing) + ' ' + str(max_length) + ' ' + 
                     method_ode + ' ' + method_interp + ' ' + filename)

        # Verify VTK file exists
        from os.path import exists
        assert exists(filename)
        
        # Check other inputs
        if startPts != None:
            assert isinstance(startPts, list)
        if direction != None:
            assert isinstance(direction, integrate_direction)
        assert isinstance(Stop_Function, types.FunctionType)
        assert( method_ode == 'RK23' or method_ode == 'RK45' or method_ode == 'DOP853')
        assert( method_interp == 'linear' or method_interp == 'nearest')
        assert isinstance( F_field, str )
        
        from vtk import vtkUnstructuredGridReader, vtkCellCenters
        from vtk.util import numpy_support as vn
        
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

        # Find the cell centers
        cellCenters = vtkCellCenters()
        cellCenters.SetInputDataObject(data)
        cellCenters.Update()
        Cpts = vn.vtk_to_numpy(cellCenters.GetOutput().GetPoints().GetData())
        x = Cpts[:,0]
        y = Cpts[:,1]
        z = Cpts[:,2]
        
        # Initialize the start points for the field line tracing, along
        # with whether we integrate forward, backwards, or both
        self.startPts = startPts
        self.direction = direction

        # Initialize storage for field lines. One field line for each
        # start point, unless we go in both directions and there are two
        # for each start point
        if( startPts != None ):
            if (direction == integrate_direction.forward or
                direction == integrate_direction.backward):
                self.fieldlines = [None]*len(startPts)
            else:
                self.fieldlines = [None]*2*len(startPts)
        
        # Initialize multitrace for tracing field lines
        self.multitrace = multitrace_cartesian_unstructured( x, y, z, Fx, Fy, Fz,
                                       Stop_Function = Stop_Function, 
                                       tol = tol, grid_spacing = grid_spacing, 
                                       max_length = max_length, 
                                       method_ode = method_ode, 
                                       method_interp = method_interp )
        
        return
    
    def setstartpoints(self, startPts=None, 
                       direction=integrate_direction.forward):
        """Set start points and direction of integration
            
        Inputs:
            startPts = list of start points for starting integration, each
                start point is nd.array with cartesian coordinates
            direction = integrate forward, backward or both directions from
                each start point, must be a member of enum integrate_direction
            
        Outputs:
            None
        """

        logging.info('Setting start points field line...')

        assert isinstance(startPts, list)
        assert len(startPts) >= 1
        assert isinstance(direction, integrate_direction)
 
        self.startPts = startPts
        self.direction = direction

        # Initialize storage for field lines. One field line for each
        # start point, unless we go in both directions and there are two
        # for each start point
        if (direction == integrate_direction.forward or
            direction == integrate_direction.backward):
            self.fieldlines = [None]*len(startPts)
        else:
            self.fieldlines = [None]*2*len(startPts)

        return
    
    def tracefieldlines(self):
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

        num_startpts = len(self.startPts)
        for i in range(num_startpts):
            # Trace field line
            if(self.direction != integrate_direction.both):
                self.fieldlines[i] = self.multitrace.trace_field_line(self.startPts[i], 
                                        self.direction == integrate_direction.forward)
            else:
                self.fieldlines[i] = self.multitrace.trace_field_line(self.startPts[i], True)
                self.fieldlines[i+num_startpts] = self.multitrace.trace_field_line(self.startPts[i], False)

        # def wrap(n):
        #     logging.info("Working on start point {}".format(n))
            
        #     # When both directions selected, put forward line at n and backward
        #     # line at n+num_pts
            
        #     if( self.direction == integrate_direction.both ):
        #         self.fieldlines[n] = self.multitrace.trace_field_line(self.startPts[n], True)
        #         self.fieldlines[n+num_startpts] = self.multitrace.trace_field_line(self.startPts[n], False)
        #     else:      
        #         self.fieldlines[n] = self.multitrace.trace_field_line(self.startPts[n], 
        #                                 self.direction == integrate_direction.forward)

        # if parallel:
        #     from joblib import Parallel, delayed
        #     import multiprocessing
        #     num_cores = multiprocessing.cpu_count()
        #     num_cores = min(num_cores, num_startpts, 20)
        #     logging.info(f'Parallel processing {num_startpts} field lines using {num_cores} cores')            
        #     Parallel(n_jobs=num_cores)(delayed(wrap)(n) for n in range(num_startpts))
        # else:
        #     logging.info(f'Serial processing {num_startpts} field lines')
        #     for n in range(num_startpts):
        #         wrap(n)

        return self.fieldlines    
