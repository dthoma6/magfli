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

   
    
    