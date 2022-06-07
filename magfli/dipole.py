#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 10:15:15 2022

@author: Dean Thomas
"""

import numpy as np
import logging

def dipole_earth_spherical(r, theta, phi):
    """Calculate earth's magnetic field in spherical coordinates based on a 
    simple dipole model
    
    https://en.wikipedia.org/wiki/Dipole_model_of_the_Earth%27s_magnetic_field
    
    Inputs: 
        r = radius of point where B field is measured in units of earth radius
        theta = colatitude (measured from earth's magnetic north pole)
        phi = angle around earth measured in x-y plane, phi = 0 along positive x axis
                            
    Outputs:
        B = B[0] = B field in r-hat direction 
            B[1] = B field in theta-hat direction
            B[2] = B field in phi-hat direction (always zero in this simple model)
        all fields in units of Tesla
    """
    assert( r > 0 )  # Avoid singularity at r = 0
    B0 = 3.12e-5
    
    B = np.zeros(3)
    B[0] = - 2 * B0 * np.cos(theta) / r**3
    B[1] = -     B0 * np.sin(theta) / r**3
    B[2] = 0
    
    return B

def dipole_earth_cartesian( X ):
    """Calculate earth's magnetic field in cartesian coordinates based on a 
    simple dipole model
    
    https://en.wikipedia.org/wiki/Dipole_model_of_the_Earth%27s_magnetic_field
    
    To develop the equations in this routine, just transform the equations from
    Wikipedia into cartesian coordinates.
    
    Inputs: 
        X = x-y-z position where B field is measured in units of earth radius.
            X[0] = x, X[1] = y, X[2] =z
        
        x is approximately toward sun, z is toward earth north magnetic pole, 
        y follows right-hand rule
                            
    Outputs:
        Bx, By, Bz = B field in x-hat, y-hat, and z-hat directions 
        All fields in units of Tesla
    """
    B0 = 3.12e-5
    
    r = np.linalg.norm( X )
    assert( r > 0 ) # avoid singularity at r = 0
      
    B = np.zeros(3)
    B[0] = - 3 * B0 * X[0] * X[2] / r**5
    B[1] = - 3 * B0 * X[1] * X[2] / r**5
    B[2] = - B0 * ( 3 * X[2]**2 - r**2 ) / r**5
    
    return B

def dipole_mag_line_spherical(theta, r0, theta0):
    """Calculate range r to field line at colatitude theta.  Field line
    starts at r0, theta0.  r and r0 are in units of earth's radius.
    
    See Willis and Young, 1987, Geophys. J.R. Astr. Soc. 89, 1011-1022
    Equation for the field lines of an axisummertic magnetic multipole
    
                  r = r1*sin(theta)**2 for dipole
    
    Inputs:
        theta = colatitude where field line wanted
        theta0 = colatitude where field line started
        
    Outputs:
        r = range in units of earth's radius to field line at theta that starts
            at r0, theta0
    """
    assert( r0 > 0 ) # avoid singularity at r0 = 0
    
    r1 = r0 / np.sin(theta0)**2
    r = r1 * np.sin(theta)**2
        
    return r

def dipole_mag_line_delta_cartesian(X, X0):
    """Calculate difference between analytic solution for field line to the
    estimated position at x,y and z. 
    
    Inputs: 
        X = x,y,z position where user estimates position of field line
        X0 = x,y,z position where field line starts in units of earth radius
        
    Outputs:
        del_r = delta radius between analytic solution for field line and x,y,z
    """
    
    # Determine radius to where field line starts (r0) and range to estimated
    # position of field line at X (r1)
    r0 = np.linalg.norm( X0 )
    r1 = np.linalg.norm( X )
    
    assert( r0 > 0 ) # avoid singularities when r = 0
    assert( r1 > 0 )
    
    # Determine angles where field line starts (theta0), and where estimated 
    # field line is estimated at x,y,z (theta1)
    theta0 = np.arccos( X0[2]/r0 )
    theta1 = np.arccos( X[2]/r1 )
    
    # Get range r_analytic to field line at theta1
    r_analytic = dipole_mag_line_spherical( theta1, r0, theta0 )
    
    # Delta between analytic range and position x,y,z
    return r1 - r_analytic

def dipole_earth_cartesian_regular_grid(Xmax, grid_spacing):
    """Create regularly spaced grid containing dipole magnetic field at x,y,z
    points. Result is a 3D grid that is centered on origin containing vector 
    values of magnetic field, B.
    
    Inputs:
        Xmax = defines bounds of grid, x => -Xmax[0] -> +Xmax[0], y => -Xmax[1]
            -> +Xmax[1], and z => -Xmax[2] -> +Xmax[2]
        grid_spacing = distance between points along x, y, or z
        
        Note: Xmax values may be changed by algorithm to avoid the origin, which
        causes a divide by zero condition when calculating the B field
        
    Outputs:
        x[], y[], z[] = magnetic field vector determined at points x,y,z
        Bx[], By[], Bz[] = magnetic field vector arrays
    """
    
    # Determine domain for x,y,z
    # Note, we want to avoid the origin, which causes a divide by zero
    # condition when calculating the B field.  So we adjust if necessary
    # by adding half the grid_spacing
    xdel = ydel = zdel =0
    if( np.mod(Xmax[0],grid_spacing) == 0 ): 
        logging.info('Modifying Xmax[0] to avoid origin')
        xdel = grid_spacing/2
    if( np.mod(Xmax[1],grid_spacing) == 0 ): 
        logging.info('Modifying Xmax[1] to avoid origin')
        ydel = grid_spacing/2
    if( np.mod(Xmax[2],grid_spacing) == 0 ): 
        logging.info('Modifying Xmax[2] to avoid origin')
        zdel = grid_spacing/2
    x = np.arange(-Xmax[0]+xdel, Xmax[0], grid_spacing)
    y = np.arange(-Xmax[1]+ydel, Xmax[1], grid_spacing)
    z = np.arange(-Xmax[2]+zdel, Xmax[2], grid_spacing)
    
    # Create mesh grid
    xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
    
    # Initialize B arrays
    shape = np.shape(xg)
    Bx = np.zeros(shape)
    By = np.zeros(shape)
    Bz = np.zeros(shape)
    
    # Determine Bx, By, Bz at each point x,y,z in mesh grid
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                Bx[i,j,k], By[i,j,k], Bz[i,j,k] = dipole_earth_cartesian([xg[i,j,k], yg[i,j,k], zg[i,j,k]])
    
    return x, y, z, Bx, By, Bz

def dipole_earth_cartesian_unstructured(Xmax, num_pts, seed=12345):
    """Create unstructured, randomly-spaced grid containing dipole magnetic 
    field at x,y,z points. Result is a 3D grid that is centered on origin 
    containing vector values of magnetic field, B.
    
    Inputs:
        Xmax = defines bounds of grid, x => -Xmax[0] -> +Xmax[0], y => -Xmax[1]
            -> +Xmax[1], and z => -Xmax[2] -> +Xmax[2]
        num_pts = number of points in unstructured data set
        seed = random generator seed (integer)
        
    Outputs:
        x[], y[], z[] = magnetic field vector determined at points x,y,z
        Bx[], By[], Bz[] = magnetic field vector arrays
    """
    
    assert( Xmax[0] > 0 and Xmax[1] > 0 and Xmax[2] > 0 )
    assert( num_pts > 1 )
    assert isinstance( seed, int )
    
    # Initialize arrays
    x = np.zeros(num_pts)
    y = np.zeros(num_pts)
    z = np.zeros(num_pts)
    Bx = np.zeros(num_pts)
    By = np.zeros(num_pts)
    Bz = np.zeros(num_pts)

    # Select random x,y,z points and the cooresponding value of Bx, By, 
    # and Bz at each point.  
    rng = np.random.default_rng(seed)
    for i in range(num_pts):
        x[i] = -Xmax[0] + 2*Xmax[0] * rng.random()
        y[i] = -Xmax[1] + 2*Xmax[1] * rng.random()
        z[i] = -Xmax[2] + 2*Xmax[2] * rng.random()
        Bx[i], By[i], Bz[i] = dipole_earth_cartesian([x[i], y[i], z[i]])
      
    return x, y, z, Bx, By, Bz

if __name__ == "__main__":
    dipole_earth_cartesian_regular_grid([5,5,5],1)