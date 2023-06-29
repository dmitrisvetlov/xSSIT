# Two species Poisson Process

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import csr_array
from scipy.sparse.linalg import expm_multiply
from scipy.stats import poisson
import timeit

from pyexpokit import expmv

numArrayDimensions = 6
numSimulations = 1
arrayDimensions = np.floor(np.logspace(1, 2, numArrayDimensions))
simDataTemplateArray = np.ones((numSimulations, numArrayDimensions))

timeExpm = np.nan * simDataTemplateArray
timeExpokit = np.nan * simDataTemplateArray
timeODE23s = np.nan * simDataTemplateArray

err1Expm = np.nan * simDataTemplateArray
err1Expokit = np.nan * simDataTemplateArray
err1ODE23s = np.nan * simDataTemplateArray
err2Expm = np.nan * simDataTemplateArray
err2Expokit = np.nan * simDataTemplateArray
err2ODE23s = np.nan * simDataTemplateArray

diff_expm_expokit = np.nan * simDataTemplateArray
diff_expm_ode23s = np.nan * simDataTemplateArray
diff_expokit_ode23s = np.nan * simDataTemplateArray

k1 = 10
k2 = 13
g1 = 1
g2 = 0.5

startPt = 1 # Starting timepoint 
numPt = 2 # Total number of timepoints

COL_AXIS = 0

for index in range(len(arrayDimensions) - 1, -1, -1):
    N = int(arrayDimensions[index]);

    A = np.zeros(((N + 1) ** 2, (N + 1) ** 2))
    C1 = np.zeros((N + 1, (N + 1) ** 2))
    C2 = np.zeros((N + 1, (N + 1) ** 2))

    for i1 in range(N):
        for i2 in range(N):
            k = i1 * (N + 1) + i2 # Do not add 1; zero-based indexing
            
            if i1 > 0:
                A[k][k] = A[k][k] - (g1 * i1)
                A[k - (N + 1)][k] = (g1 * i1)
        
            if i2 > 0:
                A[k][k] = A[k][k]- (g2 * i2)
                A[k - 1][k] = (g2 * i2)
            
            if i1 < N:
                A[k][k] = A[k][k] - k1
                A[k + (N + 1)][k] = k1
            
            if i2 < N:
                A[k][k] = A[k][k] - k2
                A[k + 1][k] = k2
            
            C1[i1][k] = 1 # Do not add 1; zero-based indexing
            C2[i2][k] = 1 # Do not add 1; zero-based indexing
        # for i2 in range(N) ...
    # for i1 in range(N) ...

    P0 = np.zeros(((N + 1) ** 2, 1))
    P0[1] = 1
    
    #t = 2.4
    
    # Convert the sparse matrices:
        
    A = csr_array(A)
    C1 = csr_array(C1)
    C2 = csr_array(C2)
    
    for simCntr in range(numSimulations):
    
        # Solve the ODE via expm_multiply and record the solution and time
        # required in the appropriate matrices. (We refer to the operation as
        # "expm," but scipy provides an expm_multiply that includes the taking
        # of the dot product of e^A and P0.) We will need to do this twice, 
        # because generated variables do not persist after timeit is executed:
        
        timeExpm[simCntr][index] = timeit.timeit(
            stmt = """expm_multiply(A, P0, start = startPt,
            stop = startPt + numPt,
            num = numPt, endpoint = False)""", number = 1, globals = globals())
        
        expAt_P = expm_multiply(A, P0, start = startPt,
                                stop = startPt + numPt,
                                num = numPt, endpoint = False)
        
        
        # Pt = expm(A*t)*P0;
        
        P1t = C1 @ expAt_P[-1,:,:]
        P2t = C2 @ expAt_P[-1,:,:]
        
        # Check the results
        
        nrange = np.arange(start = 0, stop = N + 1) # To include N
        
        lam1 = k1 / g1 * (1 - np.exp(-g1 * numPt))
        y1 = poisson.pmf(nrange, lam1)
        err1Expm[simCntr][index] = np.sum(np.abs(y1 - P1t[:,-1]))
        
        lam2 = k2 / g2 * (1 - np.exp(-g2 * numPt))
        y2 = poisson.pmf(nrange, lam2)
        err2Expm[simCntr][index] = np.sum(np.abs(y2 - P2t[:,-1]))

        # Solve the ODE via a modified version of expokit and record the 
        # solution and time required in the appropriate matrices. We will need
        # to do this twice, because generated variables do not persist after 
        # timeit is executed:
        
        timeExpokit[simCntr][index] = timeit.timeit(
            stmt = "expmv(t = numPt, A = A, v = P0)", number = 1,
            globals = globals())
        
        PfExpokit = expmv(t = numPt, A = A, v = P0)
        
        P1t = C1 @ PfExpokit
        P2t = C2 @ PfExpokit
        err1Expokit[simCntr][index] = np.sum(np.abs(y1 - P1t))
        err2Expokit[simCntr][index] = np.sum(np.abs(y2 - P2t))
        
        # Solve the ODE via the solve_ivp function provided by SciPy. Record
        # the solution and time required in the appropriate matrices. We will 
        # need to do this twice, because generated variables do not persist 
        # after timeit is executed:
            
        def rhs(t, y): return A @ y
        t_span = [0, numPt]
        y0 = P0.flatten()
        method = 'RK23'
        t_eval = [0, numPt / 2, numPt]
        rtol = 1e-8
        atol = 1e-10
        # The SciPy documentation states that vectorization will slow down
        # finite-difference approximation for all methods other than Radau and
        # BDF...
        vectorized = False
        jac = A
        
        stmt = """solve_ivp(fun = rhs, t_span = t_span, y0 = y0,
        method = method, t_eval = t_eval, vectorized = vectorized,
        rtol = rtol, atol = atol, jac = A)"""
        
        timeODE23s[simCntr][index] = timeit.timeit(
            stmt = stmt, number = 1, globals = globals())
        
        sol = solve_ivp(fun = rhs, t_span = t_span, y0 = y0, method = method,
                        t_eval = t_eval, vectorized = vectorized,
                        rtol = rtol, atol = atol, jac = A)
        
        tExport = sol.t
        yout = sol.y
        
        P1t = C1 @ yout
        P2t = C2 @ yout
        err1ODE23s[simCntr][index] = np.sum(np.abs(y1 - P1t[:,-1]))
        err2ODE23s[simCntr][index] = np.sum(np.abs(y2 - P2t[:,-1]))
    
        diff_expm_expokit[simCntr][index] = np.sum(
            np.abs(PfExpokit - (expAt_P[:][-1])[:,-1]))
        diff_expm_ode23s[simCntr][index] = np.sum(
            np.abs((expAt_P[:][-1])[:,-1] - (yout.T)[-1][:]))
        diff_expokit_ode23s[simCntr][index] = np.sum(
            np.abs(PfExpokit - ((yout.T)[-1][:])))
    # for simCntr in range(numSimulations) ...
# for index in range(len(arrayDimensions) - 1, -1, -1) ...
    
# We will now generate figures comparing the running times and results of
# the various ODE solvers tested.

fig, axs = plt.subplots(3, 1)

# Running times

meanTimeExpm = np.mean(timeExpm, axis = COL_AXIS)
meanTimeExpokit = np.mean(timeExpokit, axis = COL_AXIS)
meanTimeODE23s = np.mean(timeODE23s, axis = COL_AXIS)

axs[0].loglog(arrayDimensions, meanTimeExpm, label = 'expm')
axs[0].loglog(arrayDimensions, meanTimeExpokit, label = 'expokit')
axs[0].loglog(arrayDimensions, meanTimeODE23s, label = 'ode23s')

axs[0].set_ylabel('Computational time')

np.savetxt('meanTimeExpm.txt', meanTimeExpm)
np.savetxt('meanTimeExpokit.txt', meanTimeExpokit)
np.savetxt('meanTimeODE23s.txt', meanTimeODE23s)

np.savetxt('arrayDimensions.txt', arrayDimensions)

# Differences versus analytic result

meanErr1Expm = np.mean(err1Expm, axis = COL_AXIS)
meanErr2Expm = np.mean(err2Expm, axis = COL_AXIS)
meanErr1Expokit = np.mean(err1Expokit, axis = COL_AXIS)
meanErr2Expokit = np.mean(err2Expokit, axis = COL_AXIS)
meanErr1ODE23s = np.mean(err1ODE23s, axis = COL_AXIS)
meanErr2ODE23s = np.mean(err2ODE23s, axis = COL_AXIS)

axs[1].loglog(arrayDimensions, meanErr1Expm, label = 'expm-1')
axs[1].loglog(arrayDimensions, meanErr2Expm, label = 'expm-2')
axs[1].loglog(arrayDimensions, meanErr1Expokit, label = 'expokit-1')
axs[1].loglog(arrayDimensions, meanErr2Expokit, label = 'expokit-2')
axs[1].loglog(arrayDimensions, meanErr1ODE23s, label = 'ode23s-1')
axs[1].loglog(arrayDimensions, meanErr2ODE23s, label = 'ode23s-2')

axs[1].set_ylabel('Difference versus analytical result')

np.savetxt('meanErr1Expm.txt', meanErr1Expm)
np.savetxt('meanErr2Expm.txt', meanErr2Expm)
np.savetxt('meanErr1Expokit.txt', meanErr1Expokit)
np.savetxt('meanErr2Expokit.txt', meanErr2Expokit)
np.savetxt('meanErr1ODE23s.txt', meanErr1ODE23s)
np.savetxt('meanErr2ODE23s.txt', meanErr2ODE23s)

# Differences between methods

meanDiffExpmExpokit = np.mean(diff_expm_expokit, axis = COL_AXIS)
meanDiffExpmODE23s = np.mean(diff_expm_ode23s, axis = COL_AXIS)
meanDiffExpokitODE23s = np.mean(diff_expokit_ode23s, axis = COL_AXIS)

axs[2].loglog(
    arrayDimensions, meanDiffExpmExpokit, label = 'diff_expm_expokit')
axs[2].loglog(arrayDimensions, meanDiffExpmODE23s, label = 'diff_expm_ode23s')
axs[2].loglog(arrayDimensions, meanDiffExpokitODE23s, label = 'diff_expokit_ode23s')

axs[2].set_xlabel('Number of states')
axs[2].set_ylabel('Difference of average results')

plt.legend(loc = 'best')

# fig.tight_layout()
plt.show()

fig.savefig('foo.pdf')
fig.savefig('foo.png')