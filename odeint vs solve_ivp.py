#!/usr/bin/env python3
# from __future__ import division, print_function

import sys
import time
import multiprocessing as mp
import numpy as np
from scipy.integrate import odeint, solve_ivp

# Comments below contain results from running doit().  To obtain the results,
# DO_ODEINT is flipped between True and False, and the num_processors is varied.

# runfile('/Users/dean/Documents/GitHub/magfli/odeint vs solve_ivp.py', wdir='/Users/dean/Documents/GitHub/magfli')
# odeint
# multiprocessing:    6.362 seconds
# serial:            16.783 seconds
# num_processes = 8, speedup = 2.64

# runfile('/Users/dean/Documents/GitHub/magfli/odeint vs solve_ivp.py', wdir='/Users/dean/Documents/GitHub/magfli')
# solve_ivp
# multiprocessing:   47.467 seconds
# serial:           142.121 seconds
# num_processes = 8, speedup = 2.99

# runfile('/Users/dean/Documents/GitHub/magfli/odeint vs solve_ivp.py', wdir='/Users/dean/Documents/GitHub/magfli')
# odeint
# multiprocessing:    6.001 seconds
# serial:            17.549 seconds
# num_processes = 6, speedup = 2.92

# runfile('/Users/dean/Documents/GitHub/magfli/odeint vs solve_ivp.py', wdir='/Users/dean/Documents/GitHub/magfli')
# solve_ivp
# multiprocessing:   46.311 seconds
# serial:           145.751 seconds
# num_processes = 6, speedup = 2.80

# runfile('/Users/dean/Documents/GitHub/magfli/odeint vs solve_ivp.py', wdir='/Users/dean/Documents/GitHub/magfli')
# odeint
# multiprocessing:    6.310 seconds
# serial:            16.871 seconds
# num_processes = 4, speedup = 2.67

# runfile('/Users/dean/Documents/GitHub/magfli/odeint vs solve_ivp.py', wdir='/Users/dean/Documents/GitHub/magfli')
# solve_ivp
# multiprocessing:   44.094 seconds
# serial:           142.006 seconds
# num_processes = 4, speedup = 3.22

# runfile('/Users/dean/Documents/GitHub/magfli/odeint vs solve_ivp.py', wdir='/Users/dean/Documents/GitHub/magfli')
# odeint
# multiprocessing:    6.997 seconds
# serial:            16.976 seconds
# num_processes = 3, speedup = 2.43

# runfile('/Users/dean/Documents/GitHub/magfli/odeint vs solve_ivp.py', wdir='/Users/dean/Documents/GitHub/magfli')
# solve_ivp
# multiprocessing:   54.427 seconds
# serial:           141.647 seconds
# num_processes = 3, speedup = 2.60

# runfile('/Users/dean/Documents/GitHub/magfli/odeint vs solve_ivp.py', wdir='/Users/dean/Documents/GitHub/magfli')
# odeint
# multiprocessing:    9.305 seconds
# serial:            16.750 seconds
# num_processes = 2, speedup = 1.80

# runfile('/Users/dean/Documents/GitHub/magfli/odeint vs solve_ivp.py', wdir='/Users/dean/Documents/GitHub/magfli')
# solve_ivp
# multiprocessing:   75.047 seconds
# serial:           142.133 seconds
# num_processes = 2, speedup = 1.89


#DO_ODEINT = True

# Code below is derived from code on:
# https://stackoverflow.com/questions/34291639/multiple-scipy-integrate-ode-instances
    
def solve(ic, DO_ODEINT):
    if( DO_ODEINT ):
        def lorenz(q, t, sigma, rho, beta):
            x, y, z = q
            return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]
    else:
        def lorenz(t, q, sigma, rho, beta):
            x, y, z = q
            return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

    #t = np.linspace(0, 200, 801)
    t = np.arange(0,200.025,0.25)
    sigma = 10.0
    rho = 28.0
    beta = 8/3
    
    if( DO_ODEINT ):
        sol = odeint(lorenz, ic, t, args=(sigma, rho, beta), rtol=1e-10, atol=1e-12)
        return sol
    else:
        sol = solve_ivp(fun=lorenz, t_span=[t[0], t[-1]], 
                        y0=ic, 
                        t_eval=t, 
                        args=(sigma, rho, beta), 
                        rtol=1e-10, atol=1e-12, method='DOP853') 
        return sol.y

def doit(DO_ODEINT, num_processes):
    #args = [( np.random.randn(1,3) ) for i in range(100)]
    num = 100
    args = [None]*num
    for i in range(num) :
        args[i] = [np.random.randn(3), DO_ODEINT]
   
    if(DO_ODEINT):
        print('odeint')
    else:
        print('solve_ivp')

    print("multiprocessing:", end='')
    tstart = time.time()
 
    p = mp.Pool(num_processes)
    mp_solutions =[]
    for result in p.starmap(solve, args):
        mp_solutions.append(result)
    tend = time.time()
    tmp = tend - tstart
    print(" %8.3f seconds" % tmp)

    print("serial:         ", end='')
    sys.stdout.flush()
    tstart = time.time()
    serial_solutions = [solve(arg[0], arg[1]) for arg in args]
    tend = time.time()
    tserial = tend - tstart
    print(" %8.3f seconds" % tserial)

    print("num_processes = %i, speedup = %.2f" % (num_processes, tserial/tmp))

    check = [(sol1 == sol2).all()
              for sol1, sol2 in zip(serial_solutions, mp_solutions)]
    if not all(check):
            print("There was at least one discrepancy in the solutions.")
            
    return tserial, tmp

# We see that clock time improvements end after 4 processors.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
# advises that solve_ivp be used for new code.  solve_ivp has more options than 
# odeint, such as a terminate event function. However, solve_ivp, as seen
# in the graph below, is signficantly slower than odeint.  
# 
# Also note, we need to be careful with which ODE solver is selected.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
# notes that some ODE solvers are not reentrant, and so are unsuitable for
# multiprocessing

def plotit():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes()
    ax.set_title('Time to Complete 100 lorenz solutions')
    ax.set_xlabel('Num of Cores')
    ax.set_ylabel('Time (sec)')
    # ax.set_xlim(-10,10)
    # ax.set_ylim(-10,10)
    
    num = (1,2,3,4,6,8)
    odeint_times = (16.783, 9.305, 6.997, 6.310, 6.001,  6.362)
    solve_ivp_times = (142.133, 75.047, 54.427, 44.094, 46.311, 47.467)

    ax.plot( num, odeint_times, color='blue', label='odeint' )
    ax.plot( num, solve_ivp_times, color='red', label='solve_ivp')
    
    ax.legend()
 
def getresults():
    num = [1,2,3,4,6,8]
    n = len(num)

    odeint_times =[None] * n
    solve_ivp_times = [None] * n
    
    j = 1
    for i in num[1::]:
        odeint_times[0], odeint_times[j] = doit(True, i)
        j = j+1
 
    j = 1
    for i in num[1::]:
        solve_ivp_times[0], solve_ivp_times[j] = doit(False, i)
        j = j+1
        
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes()
    ax.set_title('Time to Complete 100 lorenz solutions')
    ax.set_xlabel('Num of Cores')
    ax.set_ylabel('Time (sec)')
    # ax.set_xlim(-10,10)
    # ax.set_ylim(-10,10)
    
    ax.plot( num, odeint_times, color='blue', label='odeint' )
    ax.plot( num, solve_ivp_times, color='red', label='solve_ivp')
    ax.legend()
    

if __name__ == "__main__":
    # getresults()
    plotit()