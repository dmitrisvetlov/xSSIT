#  [w, err, hump] = mexpv( t, A, v, tol, m )
#  MEXPV computes an approximation of w = exp(t*A)*v using Krylov
#  subspace projection techniques. This is a customised version for
#  Markov Chains. This means that a check is done within this code to
#  ensure that the resulting vector w is a probability vector, i.e.,
#  w must have all its components in [0,1], with sum equal to 1.
#  This check is done at some expense and the user may try EXPV
#  which is cheaper since it ignores probability constraints.
#
#  IMPORTANT: The check assumes that the transition rate matrix Q
#             satisfies Qe = 0, where e = (1,...,1)'. Don't use MEXPV
#             if this condition does not hold. Use EXPV instead.
#             MEXPV/EXPV require A = Q', i.e., the TRANSPOSE of Q.
#             Failure to remember this leads to wrong results.
#
#
#  MEXPV does not compute the matrix exponential in isolation but
#  instead, it computes directly the action of the exponential operator
#  on the operand vector. This way of doing so allows for addressing
#  large sparse problems. The matrix under consideration interacts only
#  via matrix-vector products (matrix-free method).
#
#  w = mexpv( t, A, v )
#  computes w = exp(t*A)*v using a default tol = 1.0e-7 and m = 30.
#
#  [w, err] = mexpv( t, A, v )
#  renders an estimate of the error on the approximation.
#
#  [w, err] = mexpv( t, A, v, tol )
#  overrides default tolerance.
#
#  [w, err] = mexpv( t, A, v, tol, m )
#  overrides default tolerance and dimension of the Krylov subspace.
#
#  [w, err, hump] = expv( t, A, v, tol, m )
#  overrides default tolerance and dimension of the Krylov subspace,
#  and renders an approximation of the `hump'.
#
#  The hump is defined as:
#          hump = max||exp(sA)||, s in [0,t]  (or s in [t,0] if t < 0).
#  It is used as a measure of the conditioning of the matrix exponential
#  problem. The matrix exponential is well-conditioned if hump = 1,
#  whereas it is poorly-conditioned if hump >> 1.  However the solution
#  can still be relatively fairly accurate even when the hump is large
#  (the hump is an upper bound), especially when the hump and
#  ||w(t)||/||v|| are of the same order of magnitude (further details in
#  reference below). Markov chains are usually well-conditioned problems.
#
#  Example:
#  --------
#    % generate a transition rate matrix
#    n = 100
#    A = rand(n)
#    for j = 1:n
#	 sumj = 0
#        for i = 1:n
#            if rand < 0.5, A(i,j) = 0 end
#            sumj = sumj + A(i,j)
#        end
#	 A(j,j) = A(j,j)-sumj
#    end
#    v = eye(n,1)
#    A = sparse(A) % invaluable for a large and sparse matrix.
#
#    tic
#    [w,err] = expv(1,A,v)
#    toc
#
#    disp('w(1:10) =') disp(w(1:10))
#    disp('err =')     disp(err)
#
#    tic
#    w_matlab = expm(full(A))*v
#    toc
#
#    disp('w_matlab(1:10) =') disp(w_matlab(1:10))
#    gap = norm(w-w_matlab)/norm(w_matlab)
#    disp('||w-w_matlab|| / ||w_matlab|| =') disp(gap)
#
#  In the above example, n could have been set to a larger value,
#  but the computation of w_matlab will be too long (feel free to
#  discard this computation).
#
#  See also EXPV, EXPOKIT.

#  Roger B. Sidje (rbs@maths.uq.edu.au)
#  EXPOKIT: Software Package for Computing Matrix Exponentials.
#  ACM - Transactions On Mathematical Software, 24(1):130-156, 1998

import numpy as np
import sys
from scipy.linalg import norm
from scipy.sparse.linalg import expm
from scipy.sparse import sparray
from fspErrorCondition import FSPErrorCondition

def mexpv_modified_2(
        t: int, A: sparray, v: np.ndarray,
        tol: float = 1.0e-7,
        m: int = None,
        N_prt: int = None, # Change by Brian Munsky
        Time_array: np.ndarray = None, # Change by Brian Munsky
        fspTol: float = None, # Change by Brian Munsky
        SINKS: np.ndarray = None,
        tNow: int = 0,
        fspErrorCondition: FSPErrorCondition = None):
    
    n = A.shape()[0] # The size of the first dimension of A
    
    # The following code regarding default values and manipulations thereof was
    # changed by Brian Munsky:
    
    if m is None:
        m = np.min([n, 30])
        
    if Time_array is None:
        if N_prt is None:
            Time_array = np.array([0, t])
            N_prt = 1
        else:
            Time_array = np.linspace(0, t, N_prt)
            
    if fspTol is None:
        fspTol = 1.0
    else:
        tol = np.min([tol, fspTol / 10])
        
    if SINKS is None:
        SINKS = np.array([])
        
    if fspErrorCondition is None:
        fspErrorCondition = FSPErrorCondition()

    Anorm = norm(A, ord = np.inf)
    mxrej = 10
    btol = 1.0e-7
    gamma = 0.9
    delta = 1.2
    mb = m
    t_out= abs(t)
    s_error = 0
    rndoff = Anorm * sys.float_info.epsilon
    
    k1 = 2
    xm = 1.0/m
    Vnorm = norm(v, ord = 2) # Euclidean (2-norm)
    beta = Vnorm
    fact = (((m + 1) / np.exp(1))^(m + 1)) * np.sqrt(2 * np.pi * (m + 1))
    t_new = (1.0 / Anorm) * ((fact * tol) / (4 * beta * Anorm)) ^ xm
    s = 10 ^ (np.floor(np.log10(t_new)) - 1)
    t_new = np.ceil(t_new / s)*s
    
    if t == 0:
        sgn = 1
    else:
        sgn = np.sign(t) 
    
    istep = 0
    
    w = v
    np.transpose()
    ye = w.T
    te = tNow
    hump = Vnorm
    
    # Changes by Brian Munsky
    Time_array_out = np.array([])
    i_prt = 0
    P_array = np.zeros((Time_array.size, w.size))
    #
    
    if tNow == Time_array[0]:
        P_array[0] = w.T
        Time_array_out = tNow
        i_prt = 1
    
    while ((tNow < t_out) and (i_prt < Time_array.size)):
        istep += 1
        t_step = np.min([Time_array[i_prt] - tNow, t_new])
        V = np.zeros((n, m + 1))
        H = np.zeros((m + 2, m + 2))
        
        V[:, 0] = (1.0 / beta) * w
        for j in range(m):
            p = A @ V[:, j]
            for i in range(j):
                H[i,j] = p.T @ V[:, i] #8.832
                p = p - H[i, j] @ V[:, i]
            
            s = norm(p, ord = 2) # Euclidean (2-norm)
            
            if s < btol:
                k1 = 0
                mb = j
                t_step = Time_array[i_prt] - tNow
                break
            
            H[j + 1, j] = s
            V[:, j + 1] = (1.0 / s) * p
        # for j in range(m) ...
        
        if k1 != 0:
            H[m + 2, m + 1] = 1
            AVnorm = norm(A @ V[:, m + 1], ord = 2) # Euclidean (2-norm)
        
        ireject = 0
        
        while (ireject <= mxrej):
            mx = mb + k1
            F = expm(sgn * t_step * H[0:mx, 0:mx])
            if k1 == 0:
                err_loc = btol
                break
            else:
                phi1 = abs(beta * F[m + 1, 1])
                phi2 = abs(beta * F[m + 2, 1] * AVnorm)
                if (phi1 > (10 * phi2)):
                    err_loc = phi2
                    xm = 1.0 / m
                elif (phi1 > phi2):
                    err_loc = (phi1 * phi2) / (phi1 - phi2)
                    xm = 1.0 / m
                else:
                    err_loc = phi1
                    xm = 1.0 / (m - 1)
            # k1 != 0 ...
            
            if err_loc <= (delta * t_step * tol):
                break
            else:
                t_step = gamma * t_step * (t_step * tol / err_loc) ^ xm
                s = 10 ^ (np.floor(np.log10(t_step)) - 1)
                t_step = np.ceil(t_step / s) * s
                if (ireject == mxrej):
                    # Changes by Brian Munsky
                    tryagain = True
                    P_lost = w[SINKS]
                    #             time_of_excess = tNow/t
                    w = np.array([])
                    err = np.array([])
                    hump = np.array([])
                    Time_array_out = np.array([])
                    P_array = np.array([])
                    return w, err, hump, Time_array_out, P_array, P_lost, \
                        tryagain, te, ye 
                    # error('The requested tolerance is too high.')
                
                ireject += 1
            # err_loc > (delta * t_step * tol) ...
        # while (ireject <= mxrej) ...
        
        mx = mb + np.max([0, k1 - 1])
        w = V[:, 0:mx - 1] @ (beta * F[0:mx - 1, 0])
        beta = norm(w, ord = 2) # Euclidean (2-norm)
        hump = np.max([hump, beta])
        
        neg = np.asarray(w < 0).nonzero()
        ineg = neg.size
        w[neg] = 0
           
        wnorm = norm(w, ord = 1)
        if ineg > 0:
            w = (1.0 / wnorm) * w
        
        roundoff = abs(1.0 - wnorm) / n
        
        tNow += t_step
        
        # Changes by Brian Munsky
        
        if np.max(w[SINKS]) * SINKS.size > fspTol*(tNow - fspErrorCondition.tInit) / np.max(Time_array - fspErrorCondition.tInit):
            P_lost = w[SINKS]
            err = np.array([])
            hump = np.array([])
            ye = w.T
            te = tNow - t_step
            P_array = P_array[1: i_prt - 1, :]
            tryagain = False
            return w, err, hump, Time_array_out, P_array, P_lost, \
                tryagain, te, ye
        else:
            ye = w.T
            te = tNow
        
        while (i_prt < Time_array.size) and (tNow >= Time_array[i_prt]):
            Time_array_out = np.vstack((Time_array_out, Time_array[i_prt]))
            P_array[i_prt, :] = w
            i_prt += 1
    
        # End of changes by Brian Munsky
        
        t_new = gamma * t_step * (t_step * tol / err_loc) ^ xm
        s = 10^(np.floor(np.log10(t_new)) - 1)
        t_new = np.ceil(t_new / s) * s
        
        err_loc = max(err_loc, roundoff, rndoff)
        s_error += err_loc
    
    # while ((tNow < t_out) and (i_prt < Time_array.size)) ...
    
    while (i_prt < Time_array.size) and (tNow >= Time_array[i_prt]):
        Time_array_out = np.vstack((Time_array_out, Time_array[i_prt]))
        P_array[i_prt, :] = w
        i_prt += 1
    
    err = s_error
    hump = hump / Vnorm
    te = np.max(Time_array)
    ye = w
    P_lost = w[SINKS]
    
    # Changes by Brian Munsky
    tryagain = False
    
    return w, err, hump, Time_array_out, P_array, P_lost, tryagain, te, ye 