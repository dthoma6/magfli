#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:56:41 2022

@author: Dean Thomas
"""

from magfli.trace_stop_funcs import trace_stop_earth
from magfli.dipole import dipole_earth_cartesian
import types
import logging

from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
import numpy as np

from enum import Enum, auto
class integrate_direction(Enum):
    forward = auto()
    backward = auto()
    both = auto()

class fieldlines_base():
    """Base class for fieldline classes.  It includes all of the procedures
    common across the fieldlines classes.  Most of the changes should be to
    __init__().  set_start_points(), trace_field_line(), trace_field_lines(), 
    field_value() and other procedures will be unchanged in most situations.
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
        assert( method_ode == 'RK23' or method_ode == 'RK45' or method_ode == 'DOP853' 
              or method_ode == 'Radau' or method_ode == 'BDF' or method_ode == 'LSODA' )
        
        # Store instance data
        self.Stop_Function = Stop_Function
        self.tol = tol
        self.grid_spacing = grid_spacing
        self.max_length = max_length
        self.method_ode = method_ode
        
        # Tell solve_ivp that it should use the Stop_Function to terminate the 
        # integration.  Our stop functions made sure that we stay within the 
        # problem's domain.  See definitions of trace_stop_... in trace_stop_funcs.py.
        self.Stop_Function.terminal = True
        
        # Initialize grid for s from 0 to max_length, solve_ivp starts at 
        # s_grid[0] and runs until the end (unless aborted by Stop_Function).
        # s is the distance down the field line.
        self.s_grid = np.arange(0,max_length+grid_spacing,grid_spacing)
                
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
                
        # Initialize storage for field lines in VTK polydata
        self.vtk_polydata = None
        
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

        logging.info('Setting start points for field line...')

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
    
    def trace_field_line(self, X0, forward):
        """Trace a single field line based on start point XO and the provided   
        field. The field is defined by a grid of points (x,y,z) at which the 
        field (Fx,Fy,Fz) is provided.  The algorithm uses solve_ivp to step 
        along the field line
        
        Inputs:
            X0 = point at which to start the field line integration
            forward = proceed forward along field line (+step) or backwards (-step)
            
        Outputs:
            field_line.y = x,y,z points along field line starting at X0
        """
       
        logging.info('Tracing field line...' + str(X0) + ' ' + str(forward))
                
        # We need to make a copy of the Stop_Function for multiprocessing
        from copy import deepcopy
        stopfunc = deepcopy(self.Stop_Function)
        stopfunc.terminal = True

        # Function dXds is used by solve_ivp to solve ODE,
        # dX/ds = F/|F|, which is used for tracing field lines.
        # X (position) and F (field) are vectors.
        # s is the distance down the field line from initial point X0
        def dXds( s, X, xmin, ymin, zmin, xmax, ymax, zmax ):
            F = np.zeros(3)
            F[0] = self.Fx_interpolate(X)
            F[1] = self.Fy_interpolate(X)
            F[2] = self.Fz_interpolate(X)
            if( not forward ):
                F = np.negative( F )
            F_mag = np.linalg.norm(F)
            return F/F_mag
        
        # Call solve_ivp to trace field line
        field_line = solve_ivp(fun=dXds, t_span=[self.s_grid[0], self.s_grid[-1]], 
                        y0=X0, t_eval=self.s_grid, rtol=self.tol,
                        events=stopfunc, method=self.method_ode,
                        args = self.bounds)
        
        
        # Check for error
        assert( field_line.status != -1 )
        
        logging.info('End field line...' + str(X0) + ' ' + str(forward))

        return field_line.y

    def trace_field_lines(self):
        """Trace multiple field lines from each start point.  Trace will go 
        forward, backward, or both directions depending on the internal value 
        of direction.  Direction set during class initialization or by call to
        setstartpoints.
            
        Inputs:
            None
            
        Outputs:
            Field lines
        """
        logging.info('Tracing field lines...')

        num_startpts = len(self.start_pts)
        
        # Loop through all the start points
        for i in range(num_startpts):
            # Trace field line
            if(self.direction != integrate_direction.both):
                self.fieldlines[i] = self.trace_field_line(self.start_pts[i], 
                                        self.direction == integrate_direction.forward)
            else:
                self.fieldlines[i] = self.trace_field_line(self.start_pts[i], True)
                self.fieldlines[i+num_startpts] = self.trace_field_line(self.start_pts[i], False)
 
        return self.fieldlines    
    
    def trace_mp_field_lines(self): 
        """Use multiprocessing to trace multiple field lines from each start 
        point.  Trace will go forward, backward, or both directions depending 
        on the internal value of direction.  Direction set during class 
        initialization or by call to setstartpoints.
            
        Inputs:
            None
            
        Outputs:
            Field lines
        """
        logging.info('Tracing field lines with multiprocessing...')

   
        # Trace field lines in using a multiprocessing pool
        import multiprocessing as mp
        pool = mp.Pool()
    
        num_startpts = len(self.start_pts)
        j = 0       # Counter for storing field lines
        
        # if-else to consider whether we are tracing in one direction or both
        if(self.direction != integrate_direction.both):
           
            # Set up arguments for call to trace_field_line
            args = [(self.start_pts[i], self.direction == integrate_direction.forward) 
                    for i in range(num_startpts)]
           
            # Create processes in pool for selected direction
            for result in pool.starmap( self.trace_field_line, args ):
                self.fieldlines[j] = result
                j = j + 1
    
        else:
            # Set up arguments for fieldlines in forward direction (True)
            args = [(self.start_pts[i], True) for i in range(num_startpts)]
    
            # Create processes in pool for forward direction
            for result in pool.starmap( self.trace_field_line, args ):
                self.fieldlines[j] = result
                j = j + 1
    
           
            # Set up arguments for fieldlines for backwards direction (False)
            args = [(self.start_pts[i], False) 
                    for i in range(num_startpts)]
    
            # Create processes in pool for backward direction
            for result in pool.starmap( self.trace_field_line, args ):
                self.fieldlines[j] = result
                j = j + 1
    
    
        return self.fieldlines

    def field_value(self, X):
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
    
    def convert_to_vtk(self):
        """Convert the field lines to VTK format.
         
        Inputs:
            None
             
        Outputs:
            Returns -1 on err, 0 on success
        """
        from vtk import vtkPoints, vtkCellArray, vtkPolyLine, vtkPolyData

        # Make sure that we have field lines to convert
        if( self.fieldlines == None ):
            logging.info('Before converting, use trace_field_lines to create field line data')
            return -1
       
        # We need to loop through all n field lines
        n = len(self.fieldlines)
        
        # Create storage for VTK data
        vtk_points = vtkPoints()
        vtk_polylines = [None] * n
        vtk_cellarray = vtkCellArray()
        k = 0
        
        # Loop through each field line
        for i in range(n):
            
            # Create a new vtkPolyLine for each field line
            vtk_polylines[i] = vtkPolyLine()
            polyline_pid = vtk_polylines[i].GetPointIds()
            
            # Loop through the points in the ith field line
            # Add the points to the vtkPoints structure
            # Note, we have one vtkPoints for all field lines.
            # In constrast, each field line has a different 
            # vtkPolyLine structure, which the point ids
            # for each field line are stored.
            m = self.fieldlines[i].shape[1]

            for j in range(m):
                # Note: VTK field lines are the transpose of the field lines
                # stored in self.fieldlines
                vtk_points.InsertNextPoint(*tuple(self.fieldlines[i][:,j].T))
                polyline_pid.InsertNextId(k+j)
            
            # Since we append the i+1 field line to the end of the i field line
            # in vtk_points, we need to keep track of k (total points after the
            # ith field lines) so that the index in vtk_polyline is correct. 
            k = k+j+1
            
            # For the ith field line add the polylines info to the ith cell
            vtk_cellarray.InsertNextCell(vtk_polylines[i])

        # Put the everything into the vtkPolyData structure
        self.vtk_polydata = vtkPolyData()
        self.vtk_polydata.SetPoints(vtk_points)
        self.vtk_polydata.SetLines(vtk_cellarray)
        
        self.vtk_polydata.Modified()

        return 0
    
    def write_vtk_to_file(self, filename = None):
        """Write field line data to VTK file.
         
        Inputs:
            None
             
        Outputs:
            Returns -1 on err, 0 on success
        """
        from vtk import vtkPolyDataWriter
        import os
       
        if( self.vtk_polydata == None ):
            logging.info('Before saving data, use create_to_vtk to create VTK data')
            return -1
        
        if( filename == None ):
            logging.info('Valid filename to store vtk_polydata must be provided')
            return -1
        
        if( not filename.endswith('.vtk') ):
           logging.info('Filename ending in .vtk expected')
           return -1
        
        path = os.path.dirname(filename)
        if( not os.path.isdir(path) ):  
            logging.info('Filename must contain a path to a valid directory')
            return -1
        
        # Everything looks OK, so write data to file
        writer = vtkPolyDataWriter()
        writer.SetInputData(self.vtk_polydata)
        writer.SetFileName(filename)
        writer.Write()
        return 0

    def display_vtk(self, earth = True):
        """Display VTK field lines.
         
        Inputs:
            None
             
        Outputs:
            Returns -1 on err, 0 on success
        """
        from vtk import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow, \
            vtkRenderWindowInteractor, vtkEarthSource, vtkSphereSource
        from vtkmodules.vtkCommonColor import vtkNamedColors
       
        if( self.vtk_polydata == None ):
            logging.info('Before displaying data, use create_to_vtk to create VTK data')
            return -1
        
        # See example code at:
        # https://kitware.github.io/vtk-examples/site/Python/GeometricObjects/CylinderExample/
        
        # The mapper is responsible for pushing the geometry into the graphics
        # library. It may also do color mapping, if scalars or other
        # attributes are defined.
        polydataMapper = vtkPolyDataMapper()
        polydataMapper.SetInputData(self.vtk_polydata)

        # The actor is a grouping mechanism: besides the geometry (mapper), it
        # also has a property, transformation matrix, and/or texture map.
        # Here we set its color and rotate it.
        polydataActor = vtkActor()
        polydataActor.SetMapper(polydataMapper)
        colors = vtkNamedColors()
        polydataActor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
        polydataActor.RotateX(-90.0)
        # polydataActor.RotateY(-45.0)

        # Create the graphics structure. The renderer renders into the render
        # window. The render window interactor captures mouse events and will
        # perform appropriate camera or actor manipulation depending on the
        # nature of the events.
        ren = vtkRenderer()
        renWin = vtkRenderWindow()
        renWin.AddRenderer(ren)
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        # Add the actors to the renderer, set the background and size
        ren.AddActor(polydataActor)
        ren.SetBackground(colors.GetColor3d("BkgColor"))
        renWin.SetSize(1000, 1000)
        renWin.SetWindowName('VTK Field Lines')

        if( earth ):
            # Add earth to image
            # Start with earth source
            earthSource = vtkEarthSource()
            earthSource.OutlineOn()
            earthSource.Update()
            earthSource.SetRadius(1.0)
            
            # Create a sphere to map the earth onto
            sphere = vtkSphereSource()
            sphere.SetThetaResolution(100)
            sphere.SetPhiResolution(100)
            sphere.SetRadius(earthSource.GetRadius())
            
            # Create mappers and actors
            earthMapper = vtkPolyDataMapper()
            earthMapper.SetInputConnection(earthSource.GetOutputPort())
            
            earthActor = vtkActor()
            earthActor.SetMapper(earthMapper)
            earthActor.GetProperty().SetColor(colors.GetColor3d('Black'))
            earthActor.RotateX(-90.0)
            # earthActor.RotateY(45.0)
            earthActor.RotateZ(240.0)
            
            sphereMapper = vtkPolyDataMapper()
            sphereMapper.SetInputConnection(sphere.GetOutputPort())
            
            sphereActor = vtkActor()
            sphereActor.SetMapper(sphereMapper)
            sphereActor.GetProperty().SetColor(colors.GetColor3d('DeepSkyBlue'))
            
            # Add the actors to the scene
            ren.AddActor(earthActor)
            ren.AddActor(sphereActor)

        # This allows the interactor to initalize itself. It has to be
        # called before an event loop.
        iren.Initialize()

        # We'll zoom out a little by accessing the camera and invoking a "Zoom"
        # method on it.
        ren.ResetCamera()
        ren.GetActiveCamera().Zoom(0.9)
        renWin.Render()

        # Start the event loop.
        iren.Start()

        return 0
        
class f_interp():
    """Class to enable the use of the function in place of an interpolator in  
    class fieldlines_cartesian_function.  All it does is store
    the info for the function to emulate an interpolator.  
    
    Note: the process is slightly inefficient.  It makes three calls to get
    the x, y, and z components, when one is sufficient.  The inefficiency is 
    ignored because it allows a lot of code to be reused.
    """
    def __init__(self, Field_Function = None, field = None):
        """Initialize f_interp
            
        Inputs:
            Field_Function = function to calculate field
            field = supply x, y, or z component of field
        Outputs:
            None
        """
        assert isinstance(Field_Function, types.FunctionType)
        assert( field == 'x' or field == 'y' or field == 'z')

        self.Field_Function = Field_Function
        self.field = field
        
        if( field == 'x' ): self.index = 0
        if( field == 'y' ): self.index = 1
        if( field == 'z' ): self.index = 2
        return
    
    def __call__(self, X):
        """Determine applicable component of field at point X. 
            
        Inputs:
            X = point (x,y,z) at which to evaluate field
        Outputs:
            Appropropriate component, as determined by self.index, of the field
                at point X
        """
        F = self.Field_Function( X )
        return F[self.index]

class fieldlines_cartesian_function(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through function that determines the field (Fx,Fy,Fz) at at given
    (x,y,z) point. Algorithm uses solve_ivp to trace the field line.  
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
        
        """Initialize fieldlines_cartesian_function
            
        Inputs:
            Xmin, Xmax = define the corners of a box bounding the domain in 
                which the field is known 
            Field_Function = function defining field vector at each x,y,z
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
        assert isinstance(Field_Function, types.FunctionType)
        assert isinstance(Stop_Function, types.FunctionType)
        assert( method_ode == 'RK23' or method_ode == 'RK45' or method_ode == 'DOP853' 
               or method_ode == 'Radau' or method_ode == 'BDF' or method_ode == 'LSODA' )
                      
        # Store instance data
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.Field_Function = Field_Function
        
        # Use function as an interpolator
        # see f_interp class above for implementation
        fx_interp = f_interp( Field_Function, 'x' )
        fy_interp = f_interp( Field_Function, 'y' )
        fz_interp = f_interp( Field_Function, 'z' )
        
        self.Fx_interpolate = fx_interp
        self.Fy_interpolate = fy_interp
        self.Fz_interpolate = fz_interp
        
        # Define box that bounds the domain of the solution.
        # Xmin and Xmax are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace_stop_funcs.py. 
        # Some of the trace_stop_... functions ignore these bounds.)
        self.bounds = Xmin + Xmax
                
        return

class fieldlines_cartesian_regular_grid(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through regular grid.  The grid is an array of points (x,y,z) 
    at which the field (Fx,Fy,Fz) is provided.  The locations of the x,y,z 
    points are regularly spaced.  Algorithm uses solve_ivp to trace the 
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
        
        """Initialize fieldlines_cartesian_regular_grid
            
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
        assert( method_interp == 'linear' or method_interp == 'nearest')
                       
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
        # and zmax. (See trace_stop_... function definitions in trace_stop_funcs.py. 
        # Some of the trace_stop_... functions ignore these bounds.)
        self.bounds = [min(x), min(y), min(z)] + [max(x), max(y), max(z)]
       
        return

class fieldlines_cartesian_unstructured(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured grid.  The grid is an array of points (x,y,z) 
    at which the field (Fx,Fy,Fz) is provided.  The locations of the x,y,z 
    points are unconstrained.  Algorithm uses solve_ivp to trace the field line.  
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
        
        """Initialize fieldlines_cartesian_unstructured
            
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
        assert( method_interp == 'linear' or method_interp == 'nearest')
                       
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
        # and zmax. (See trace_stop_... function definitions in trace_stop_funcs.py. 
        # Some of the trace_stop_... functions ignore these bounds.)
        self.bounds = [min(x), min(y), min(z)] + [max(x), max(y), max(z)]
        
        return

class fieldlines_cartesian_unstructured_BATSRUSfile(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured grid stored in a BATSRUS file.  The grid is an
    array of points (x,y,z) at which the field (Fx,Fy,Fz) is provided.  
    The locations of the x,y,z points are unconstrained.  Algorithm uses 
    solve_ivp to trace the field line.  
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

        # Read BATSRUS file
        import swmfio
        batsrus = swmfio.read_batsrus(filename)
        assert( batsrus != None )
           
        # Extract x, y, z and Fx, Fy, Fz from file
        var_dict = dict(batsrus.varidx)
        self.x = batsrus.data_arr[:,var_dict[x_field]][:]
        self.y = batsrus.data_arr[:,var_dict[y_field]][:]
        self.z = batsrus.data_arr[:,var_dict[z_field]][:]
            
        self.Fx = batsrus.data_arr[:,var_dict[Fx_field]][:]
        self.Fy = batsrus.data_arr[:,var_dict[Fy_field]][:]
        self.Fz = batsrus.data_arr[:,var_dict[Fz_field]][:]
            
        # Create interpolators for unstructured data
        # Use list(zip(...)) to convert x,y,z's => (x0,y0,z0), (x1,y1,z1), ...
        xyz = list(zip(self.x, self.y, self.z))
        if( method_interp == 'linear'):
            self.Fx_interpolate = LinearNDInterpolator( xyz, self.Fx )
            self.Fy_interpolate = LinearNDInterpolator( xyz, self.Fy )
            self.Fz_interpolate = LinearNDInterpolator( xyz, self.Fz )
        if( method_interp == 'nearest'):
            self.Fx_interpolate = NearestNDInterpolator( xyz, self.Fx )
            self.Fy_interpolate = NearestNDInterpolator( xyz, self.Fy )
            self.Fz_interpolate = NearestNDInterpolator( xyz, self.Fz )
 
        # Define box that bounds the domain of the solution.
        # min and max are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace_stop_funcs.py. 
        # Some of the trace_stop_... functions ignore these bounds.)
        self.bounds = [min(self.x), min(self.y), min(self.z)] + \
                    [max(self.x), max(self.y), max(self.z)]
        
        return

class b_interp():
    """Class to enable the use of the swmfio interpolator in class 
    fieldlines_cartesian_unstructured_swmfio_BATSRUSfile.  All it does is store
    the info for the interpolator.
    """
    def __init__(self, batsrus = None, field = None):
        """Initialize f_interp
            
        Inputs:
            Field_Function = function to calculate field
            field = supply x, y, or z component of field
        Outputs:
            None
        """
        assert( batsrus != None )
        assert( field != None )

        self.batsrus = batsrus
        self.field = field
        return
    
    def __call__(self, X):
        """interpolate to determine field at point X
            
        Inputs:
            X = point (x,y,z) at which to evaluate field
        Outputs:
            Appropriate component, as determined by self.field, of the field
                at point X
        """
        return self.batsrus.interpolate( X, self.field )
    
class fieldlines_cartesian_unstructured_swmfio_BATSRUSfile(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured grid stored in a BATSRUS file.  The grid is an
    array of points (x,y,z) at which the field (Fx,Fy,Fz) is provided.  
    The locations of the x,y,z points are unconstrained.  Algorithm uses 
    solve_ivp to trace the field line.  This class uses the swmfio interpolator
    rather than the solve_ivp interpolator used in other routines.
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

        # Read BATSRUS file
        import swmfio
        batsrus = swmfio.read_batsrus(filename)
        assert( batsrus != None )
           
        # Extract x, y, z and Fx, Fy, Fz from file
        var_dict = dict(batsrus.varidx)
        self.x = batsrus.data_arr[:,var_dict[x_field]][:]
        self.y = batsrus.data_arr[:,var_dict[y_field]][:]
        self.z = batsrus.data_arr[:,var_dict[z_field]][:]
            
        self.Fx = batsrus.data_arr[:,var_dict[Fx_field]][:]
        self.Fy = batsrus.data_arr[:,var_dict[Fy_field]][:]
        self.Fz = batsrus.data_arr[:,var_dict[Fz_field]][:]

        # Use swmfio interpolator
        # see b_interp class above for implementation
        bx_interp = b_interp( batsrus, Fx_field )
        by_interp = b_interp( batsrus, Fy_field )
        bz_interp = b_interp( batsrus, Fz_field )
        
        self.Fx_interpolate = bx_interp
        self.Fy_interpolate = by_interp
        self.Fz_interpolate = bz_interp

        # Store instance data
        self.batsrus = batsrus
        self.Fx_field = Fx_field
        self.Fy_field = Fy_field
        self.Fz_field = Fz_field        
        
        # Define box that bounds the domain of the solution.
        # min and max are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace_stop_funcs.py. 
        # Some of the trace_stop_... functions ignore these bounds.)
        self.bounds = [min(self.x), min(self.y), min(self.z)] + \
                    [max(self.x), max(self.y), max(self.z)]
               
        return

class fieldlines_cartesian_unstructured_VTK(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through a VTK unstructured grid (vtkUnstructuredGrid).  The grid is
    an array of points (x,y,z) at which the field (Fx,Fy,Fz) is provided.  
    The locations of the x,y,z points are unconstrained.  Algorithm uses 
    solve_ivp to trace the field line.  
    """

    def __init__(self, vtkData = None,
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
        
        """Initialize fieldlines_cartesian_unstructured_VTK
            
        Inputs:
            vtkData = vtkUnstructured Grid that includes x,y,z grid points and 
                field F at center of cells
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
            cell_centers = a string defining which cell array contains the 
                cell centers.  This string is optional.
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
                      method_ode + ' ' + method_interp)

        # Store input variable
        assert( vtkData is not None )
        self.vtkData = vtkData

        # Check other inputs
        assert( method_interp == 'linear' or method_interp == 'nearest')
        assert isinstance( F_field, str )
        assert (isinstance( cell_centers, str) or (cell_centers is None))
        
        from vtk import vtkCellCenters
        from vtk.util import numpy_support as vn
        
        # Note, the x,y,z points in the unstructured grid are offset from
        # the center of the cells where the field is defined.

        # The field F at each cell center
        F = vn.vtk_to_numpy(vtkData.GetCellData().GetArray(F_field))
        self.Fx = F[:,0]
        self.Fy = F[:,1]
        self.Fz = F[:,2]
        
        # We need the x,y,z locations of the cell centers.
        # If Cell_centers is str, we read them from the vtkData.
        # If Cell_centers is None, we calculate them via VTK.  
        if( isinstance( cell_centers, str ) ):
            C = vn.vtk_to_numpy(vtkData.GetCellData().GetArray(cell_centers))
            self.x = C[:,0]
            self.y = C[:,1]
            self.z = C[:,2]
        else:
            cellCenters = vtkCellCenters()
            cellCenters.SetInputDataObject(vtkData)
            cellCenters.Update()
            Cpts = vn.vtk_to_numpy(cellCenters.GetOutput().GetPoints().GetData())
            self.x = Cpts[:,0]
            self.y = Cpts[:,1]
            self.z = Cpts[:,2]
                
        # Create interpolators for unstructured data
        # Use list(zip(...)) to convert x,y,z's => (x0,y0,z0), (x1,y1,z1), ...
        xyz = list(zip(self.x, self.y, self.z))
        if( method_interp == 'linear'):
            self.Fx_interpolate = LinearNDInterpolator( xyz, self.Fx )
            self.Fy_interpolate = LinearNDInterpolator( xyz, self.Fy )
            self.Fz_interpolate = LinearNDInterpolator( xyz, self.Fz )
        if( method_interp == 'nearest'):
            self.Fx_interpolate = NearestNDInterpolator( xyz, self.Fx )
            self.Fy_interpolate = NearestNDInterpolator( xyz, self.Fy )
            self.Fz_interpolate = NearestNDInterpolator( xyz, self.Fz )
 
        # Define box that bounds the domain of the solution.
        # min and max are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace_stop_funcs.py. 
        # Some of the trace_stop_... functions ignore these bounds.)
        self.bounds = [min(self.x), min(self.y), min(self.z)] + \
                    [max(self.x), max(self.y), max(self.z)]
        
        return

class fieldlines_cartesian_unstructured_VTKfile(fieldlines_base):
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured grid stored in a VTK file.  The grid is an
    array of points (x,y,z) at which the field (Fx,Fy,Fz) is provided.  
    The locations of the x,y,z points are unconstrained.  Algorithm uses 
    solve_ivp to trace the field line.  
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
            cell_centers = a string defining which cell array contains the 
                cell centers.  This string is optional.
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
        
        logging.info('Initializing fieldlines for unstructured VTK grid (file): ' + 
                     str(tol) + ' ' + str(grid_spacing) + ' ' + str(max_length) + ' ' + 
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

        # Note, the x,y,z points in the unstructured grid are offset from
        # the center of the cells where the field is defined.
        
        # The field F at each cell center
        F = vn.vtk_to_numpy(data.GetCellData().GetArray(F_field))
        self.Fx = F[:,0]
        self.Fy = F[:,1]
        self.Fz = F[:,2]
        
        # We need the x,y,z locations of the cell centers.
        # If Cell_centers is str, we read them from the file.
        # If Cell_centers is None, we calculate them via VTK.  
        if( isinstance( cell_centers, str ) ):
            C = vn.vtk_to_numpy(data.GetCellData().GetArray(cell_centers))
            self.x = C[:,0]
            self.y = C[:,1]
            self.z = C[:,2]
        else:
            cellCenters = vtkCellCenters()
            cellCenters.SetInputDataObject(data)
            cellCenters.Update()
            Cpts = vn.vtk_to_numpy(cellCenters.GetOutput().GetPoints().GetData())
            self.x = Cpts[:,0]
            self.y = Cpts[:,1]
            self.z = Cpts[:,2]
                
        # Create interpolators for unstructured data
        # Use list(zip(...)) to convert x,y,z's => (x0,y0,z0), (x1,y1,z1), ...
        xyz = list(zip(self.x, self.y, self.z))
        if( method_interp == 'linear'):
            self.Fx_interpolate = LinearNDInterpolator( xyz, self.Fx )
            self.Fy_interpolate = LinearNDInterpolator( xyz, self.Fy )
            self.Fz_interpolate = LinearNDInterpolator( xyz, self.Fz )
        if( method_interp == 'nearest'):
            self.Fx_interpolate = NearestNDInterpolator( xyz, self.Fx )
            self.Fy_interpolate = NearestNDInterpolator( xyz, self.Fy )
            self.Fz_interpolate = NearestNDInterpolator( xyz, self.Fz )
 
        # Define box that bounds the domain of the solution.
        # min and max are opposite corners of box.  The bounds
        # are given to Stop_function as xmin, ymin, zmin, xmax, ymax, 
        # and zmax. (See trace_stop_... function definitions in trace_stop_funcs.py. 
        # Some of the trace_stop_... functions ignore these bounds.)
        self.bounds = [min(self.x), min(self.y), min(self.z)] + \
                    [max(self.x), max(self.y), max(self.z)]
        
        return
        
class fieldlines_cartesian_unstructured_paraview_VTKfile():
    """Trace multiple field lines through the provided field.  The field is 
    defined through unstructured grid stored in a VTK file.  The grid is an
    array of points (x,y,z) at which the field (Fx,Fy,Fz) is provided.  
    The locations of the x,y,z points are unconstrained.  Algorithm uses 
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

        # Pull the field lines from streamtracer so that we can store them
        # in self.fieldlines (np.arrays) like we did in the other field line
        # tracers
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
