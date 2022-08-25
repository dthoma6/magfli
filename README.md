# magfli
MAGnetic Field LIne (magfli) tracer

While this package has the name MAGnetic Field LIne tracer, there is very little in it that is specific to magnetic field lines, it is generally applicable to tracing field lines for any field.

fieldlines.py contains the functions that trace field lines where the field is defined by:
- a cartesian function that provides the vector field at any given x,y,z point
- a regular grid where the vector field is provided at specific x,y,z points (see scipy solve_ivp for details on regular grid requirements)
- an unstructured grid where the vector field is provided at specific x,y,z points (see scipy solve_ivp)
- an unstructured grid in a BATSRUS file (see Space Weather Modeling Framework for details)
- an unstructured grid in a BATSRUS file with the swmfio interpolation function
- an unstructured grid provided via a vtkUnstructuredGrid (see Visualization  Tookkit (VTK) for details)
- an unstructured grid provided via a VTK file
- an unstructured grid provided via a VTK file, but processed using paraview.simple's StreamTracer algorithm.

Except for the two cases using either the swmfio interpolator or the StreamTracer algorithm, the field lines are traced using scipy interpolators and solve_ivp.  See fieldline_base.trace_field_line in fieldlines.py for details on the implementation of the solve_ivp solution.

trace_stop_funcs.py contains functions used to terminate the field line tracing where the field line reaches specific constrains.  For example, a field line can be terminated when it reaches a domain boundary.  See the file for details.

In our testing, we observed that one of the scipy interpolators would fail on large datasets.  For unstructured grids, scipy provides a nearest neighbor and a linear interpolator.  On some large datasets, the linear interpolator failed to initialize.

Demo_traces.py shows how to trace a single field line using either a function, a regularly spaced grid, or a unstructured filled with a random array of points.  In all three cases, the field is a simple dipole model of the earth's magnetic field.  See dipole.py for details on the field.

Demo_fieldlines.py shows how to trace multiple field lines.  It uses the examples above plus more complex magnetic fields stored in BATSRTUS and VTK files.

The tests folder contains tests of the basic algorithms to verify the installation.

The compare_methods..., the timing..., and the visual validation.ipynb files contain information from our tests of the algorithms.  See them for details.
