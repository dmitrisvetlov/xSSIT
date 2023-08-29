# Two species Poisson Process

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import csr_array
from scipy.stats import poisson
import timeit

solversToUse = ['Radau', 'BDF', 'LSODA']
# The SciPy documentation states that vectorization will slow down
# finite-difference approximation for all methods other than Radau and
# BDF...
useVectorization = [True, True, False]

numSolvers = len(solversToUse)

numArrayDimensions = 6
numSimulations = 5
arrayDimensions = np.floor(np.logspace(1, 2, numArrayDimensions)) # 1, 2
 
# We will only construct the input arrays once for all solvers:
simConstrTemplateArray = np.ones((numSimulations, numArrayDimensions))
simDataTemplateArray = np.ones(
    (numSolvers, numSimulations, numArrayDimensions))

timeODEsolver = np.nan * simDataTemplateArray
timeArrayConstruction = np.nan * simConstrTemplateArray

err1ODEsolver = np.nan * simDataTemplateArray
err2ODEsolver = np.nan * simDataTemplateArray

k10 = 5
k11 = 2
om1 = 1
k20 = 5
k21 = 2
om2 = 1

g1 = 1
g2 = 1

def k1(t):
    return k10 + k11 * np.sin(om1 * t)

def k2(t):
    return k20 + k21 * np.sin(om2 * t)

startPt = 1 # Starting timepoint 
numPt = 2 # Total number of timepoints

COL_AXIS = 0

def construct_arrays(N: int):
    A0 = np.zeros(((N + 1) ** 2, (N + 1) ** 2))
    A1 = np.zeros(((N + 1) ** 2, (N + 1) ** 2))
    A2 = np.zeros(((N + 1) ** 2, (N + 1) ** 2))
    C1 = np.zeros((N + 1, (N + 1) ** 2))
    C2 = np.zeros((N + 1, (N + 1) ** 2))

    # Evaluate all (i1, i2) pairs from (0, 0) through (N, N)
    for i1 in range(N + 1):
        for i2 in range(N + 1):
            k = i1 * (N + 1) + i2 # Do not add 1; zero-based indexing
            
            if i1 > 0:
                A0[k][k] = A0[k][k] - (g1 * i1)
                A0[k - (N + 1)][k] = (g1 * i1)
        
            if i2 > 0:
                A0[k][k] = A0[k][k] - (g2 * i2)
                A0[k - 1][k] = (g2 * i2)
            
            if i1 < N:
                A1[k][k] = A1[k][k] - 1
                A1[k + (N + 1)][k] = 1
            
            if i2 < N:
                A2[k][k] = A2[k][k] - 1
                A2[k + 1][k] = 1
            
            C1[i1][k] = 1 # Do not add 1; zero-based indexing
            C2[i2][k] = 1 # Do not add 1; zero-based indexing
        # for i2 in range(N + 1) ...
    # for i1 in range(N + 1) ...

    P0 = np.zeros(((N + 1) ** 2, 1))
    P0[1] = 1
    
    #t = 2.4
    
    # Convert the sparse matrices:
        
    A0 = csr_array(A0)
    A1 = csr_array(A1)
    A2 = csr_array(A2)
    C1 = csr_array(C1)
    C2 = csr_array(C2)
    
    # A0dense = A0.todense()
    # A1dense = A1.todense()
    # A2dense = A2.toarray()
    
    return A0, A1, A2, C1, C2, P0

for index in range(0, len(arrayDimensions), 1): # range(len(arrayDimensions) - 1, -1, -1):
    N = int(arrayDimensions[index]);

    for simCntr in range(numSimulations):
        
        timeArrayConstruction[simCntr][index] = timeit.timeit(
            stmt = 'construct_arrays(N)', number = 1, globals = globals())
        
        A0, A1, A2, C1, C2, P0 = construct_arrays(N)

        # Solve the ODE via the solve_ivp function provided by SciPy. Record
        # the solution and time required in the appropriate matrices. We will 
        # need to do this twice, because generated variables do not persist 
        # after timeit is executed:
            
        def jac(t, y): return (A0 + (k1(t) * A1) + (k2(t) * A2))    
        def rhs(t, y): return jac(t, y) @ y
        
        t_span = [0, numPt]
        y0 = P0.flatten()
        t_eval = [0, numPt / 2, numPt]
        rtol = 1e-8
        atol = 1e-10
        
        for solverCntr in range(numSolvers):
            method = solversToUse[solverCntr]
            vectorized = useVectorization[solverCntr]
            
            stmt = """solve_ivp(fun = rhs, t_span = t_span, y0 = y0,
            method = method, t_eval = t_eval, vectorized = vectorized,
            rtol = rtol, atol = atol, jac = jac)"""
            
            timeODEsolver[solverCntr][simCntr][index] = timeit.timeit(
                stmt = stmt, number = 1, globals = globals())
            
            sol = solve_ivp(fun = rhs, t_span = t_span, y0 = y0, method = method,
                            t_eval = t_eval, vectorized = vectorized,
                            rtol = rtol, atol = atol, jac = jac)
            
            tExport = sol.t
            yout = sol.y
            
            # Check the results
            
            nrange = np.arange(start = 0, stop = N + 1) # To include N
            
            P1t = C1 @ yout
            P2t = C2 @ yout
            
            lam1 = (k10 / g1 * (1 - np.exp(-g1 * numPt)) +
                    (k11 * om1 * np.exp(-g1 * numPt)) / (g1 ** 2 + om1 ** 2) -
                    (k11 * (om1 * np.cos(om1 * numPt) -
                            g1 * np.sin(om1 * numPt))) / (g1 ** 2 + om1 ** 2))
            y1 = poisson.pmf(nrange, lam1)
            
            lam2 = (k20 / g2 * (1 - np.exp(-g2 * numPt)) +
                    (k21 * om2 * np.exp(-g2 * numPt)) / (g2 ** 2 + om2 ** 2) -
                    (k21 * (om2 * np.cos(om2 * numPt) -
                            g2 * np.sin(om2 * numPt))) / (g2 ** 2 + om2 ** 2))
            y2 = poisson.pmf(nrange, lam2)
            
            err1ODEsolver[solverCntr][simCntr][index] = np.sum(np.abs(y1 - P1t[:,-1]))
            err2ODEsolver[solverCntr][simCntr][index] = np.sum(np.abs(y2 - P2t[:,-1]))
        
        # for solverCntr in range(numSolvers) ...
    # for simCntr in range(numSimulations) ...
# for index in range(len(arrayDimensions) - 1, -1, -1) ...

# We will now generate figures comparing the running times and results of
# the various ODE solvers tested.

# Running times per ODE solver

fig1 = plt.figure(num = 1)

ax = fig1.subplots()

for solverCntr in range(numSolvers):
    curSolverName = solversToUse[solverCntr]
    curSolverTimeStats = np.mean(timeODEsolver[solverCntr], axis = COL_AXIS)
    
    np.savetxt('meanTimeODEsolver_%s.py.txt' % curSolverName, curSolverTimeStats)    
    
    ax.plot(arrayDimensions, curSolverTimeStats, label = ('ODEsolver-%s-py' % curSolverName))
# for solverCntr in range(numSolvers) ...    

meanTimeArrayConstruction = np.mean(timeArrayConstruction, axis = COL_AXIS)

meanTimeODE15sMatlab = np.loadtxt('timeode15s_TV.m.txt', delimiter=',')
meanTimeArrayConstructionMatlab = np.loadtxt(
    'timeConstruction_TV.m.txt', delimiter=',')
np.savetxt('arrayDimensions.py.txt', arrayDimensions)

ax.plot(arrayDimensions, meanTimeODE15sMatlab, label = 'ode15s-mat')
ax.plot(arrayDimensions, meanTimeArrayConstruction, label = 'arrays-py')
ax.plot(arrayDimensions, meanTimeArrayConstructionMatlab, label = 'arrays-mat')

ax.set_xlabel('Number of states')
ax.set_ylabel('Computational time')

plt.xscale('log')
plt.yscale('log')

plt.legend(loc = 'best')

plt.show()

fig1.savefig('times-ODEsolver.png')

# Differences versus analytic result per ODE solver

fig2 = plt.figure(num = 2)

ax = fig2.subplots()

for solverCntr in range(numSolvers):
    curSolverName = solversToUse[solverCntr]
    curSolverErr1 = np.mean(err1ODEsolver[solverCntr], axis = COL_AXIS)
    curSolverErr2 = np.mean(err2ODEsolver[solverCntr], axis = COL_AXIS)
    
    np.savetxt('meanErr1ODEsolver_%s.py.txt' % curSolverName, curSolverErr1)
    np.savetxt('meanErr2ODEsolver_%s.py.txt' % curSolverName, curSolverErr2)    
    
    ax.plot(arrayDimensions, curSolverErr1, label = ('ODEsolver-err1-%s' % curSolverName))
    ax.plot(arrayDimensions, curSolverErr2, label = ('ODEsolver-err2-%s' % curSolverName))
# for solverCntr in range(numSolvers) ...

err1ODE15sMatlab = np.loadtxt('err1_TV.m.txt', delimiter=',')
err2ODE15sMatlab = np.loadtxt('err2_TV.m.txt', delimiter=',')

ax.plot(arrayDimensions, err1ODE15sMatlab, label = ('err1-ode15s'))
ax.plot(arrayDimensions, err2ODE15sMatlab, label = ('err2-ode15s'))

ax.set_xlabel('Number of states')
ax.set_ylabel('Difference versus analytical result')

plt.xscale('log')
plt.yscale('log')

plt.legend(loc = 'best')

plt.show()

fig2.savefig('versus_analytical.png')

# Now, separate the err1 and err2 across different plots to better visualize
# the discrepancies

fig3 = plt.figure(num = 3)

ax = fig3.subplots()

for solverCntr in range(numSolvers):
    curSolverName = solversToUse[solverCntr]
    curSolverErr1 = np.mean(err1ODEsolver[solverCntr], axis = COL_AXIS)
        
    ax.plot(arrayDimensions, curSolverErr1, label = ('ODEsolver-err1-%s' % curSolverName))
# for solverCntr in range(numSolvers) ...

ax.plot(arrayDimensions, err1ODE15sMatlab, label = ('err1-ode15s'))

ax.set_xlabel('Number of states')
ax.set_ylabel('Difference versus analytical result')

plt.xscale('log')
plt.yscale('log')

plt.legend(loc = 'best')

plt.show()

fig3.savefig('versus_analytical_err1.png')

fig4 = plt.figure(num = 4)

ax = fig4.subplots()

for solverCntr in range(numSolvers):
    curSolverName = solversToUse[solverCntr]
    curSolverErr2 = np.mean(err2ODEsolver[solverCntr], axis = COL_AXIS)
        
    ax.plot(arrayDimensions, curSolverErr2, label = ('ODEsolver-err2-%s' % curSolverName))
# for solverCntr in range(numSolvers) ...

ax.plot(arrayDimensions, err2ODE15sMatlab, label = ('err2-ode15s'))

ax.set_xlabel('Number of states')
ax.set_ylabel('Difference versus analytical result')

plt.xscale('log')
plt.yscale('log')

plt.legend(loc = 'best')

plt.show()

fig4.savefig('versus_analytical_err2.png')