# Two species Poisson Process

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import savemat
from scipy.sparse import csr_array
from scipy.stats import poisson

solversToUse = ['Radau', 'BDF', 'LSODA']
# The SciPy documentation states that vectorization will slow down
# finite-difference approximation for all methods other than Radau and
# BDF...
useVectorization = [True, True, False]

numSolvers = len(solversToUse)

numArrayDimensions = 6
numSimulations = 5
arrayDimensions = np.floor(np.logspace(1, 2, numArrayDimensions)) # 1, 2

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

construct_arrays_cntr = 0

def construct_arrays(N: int, call_cntr: int):
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
    
    # Export the dense matrices
    
    mdic = {"A0": A0, "A1": A1, "A2": A2, "C1": C1, "C2": C2}
    savemat("construct_arrays_dense_%d" % call_cntr, mdic)
    
    # Convert the sparse matrices:
        
    A0 = csr_array(A0)
    A1 = csr_array(A1)
    A2 = csr_array(A2)
    C1 = csr_array(C1)
    C2 = csr_array(C2)
    
    # A0dense = A0.todense()
    # A1dense = A1.todense()
    # A2dense = A2.toarray()
    
    # Export the sparse matrices
    
    mdic = {"A0": A0, "A1": A1, "A2": A2, "C1": C1, "C2": C2}
    savemat("construct_arrays_sparse_%d" % call_cntr, mdic)
    
    return A0, A1, A2, C1, C2, P0

N = 10

construct_arrays_cntr = 0
    
A0, A1, A2, C1, C2, P0 = construct_arrays(N, construct_arrays_cntr)

construct_arrays_cntr += 1

# Solve the ODE via the solve_ivp function provided by SciPy. Record
# the solution and time required in the appropriate matrices. We will 
# need to do this twice, because generated variables do not persist 
# after timeit is executed:
    
def jac(t, y):
    jac = A0 + (k1(t) * A1) + (k2(t) * A2)
    jac_dic = {"Atemp": jac}
    savemat("jac_%f" % t, jac_dic)
    return jac    

def rhs(t, y):
    rhs = jac(t, y) @ y
    rhs_dic = {"rhs": rhs}
    savemat("rhs_%f" % t, rhs_dic)
    return rhs

t_span = [0, numPt]
y0 = P0.flatten()
t_eval = [0, numPt / 2, numPt]
rtol = 1e-8
atol = 1e-10

method = 'LSODA'
vectorized = False

stmt = """solve_ivp(fun = rhs, t_span = t_span, y0 = y0,
method = method, t_eval = t_eval, vectorized = vectorized,
rtol = rtol, atol = atol, jac = jac)"""
    
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