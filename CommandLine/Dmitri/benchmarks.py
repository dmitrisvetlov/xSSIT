import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_array, diags, sparray
from scipy.sparse.linalg import expm_multiply
import timeit

from fspErrorCondition import FSPErrorCondition
import mexpv_modified_2

def get_threshold(array1D: np.ndarray):
    sortedArray = np.sort(array1D)
    if len(sortedArray) < 5:
        return sortedArray[-1 * len(sortedArray)]
    else:
        return sortedArray[-5]
    
def test_mexpv(t: int, A: sparray, v: np.ndarray):
    m = 15
    tryAgain = True
    Time_array = np.array([0, t])
    #     fspTol = fspErrorCondition.fspTol;
    #     nSinks = fspErrorCondition.nSinks;
    
    fspErrorCondition = FSPErrorCondition()
    
    while tryAgain:
        w, err, hump, Time_array_out, P_array, P_lost, tryAgain, \
            te, ye = mexpv_modified_2(t = t, A = A, v = P0,
                                      tol = 1e-8, m = m,
                                      N_prt = None,
                                      Time_array = Time_array,
                                      fspTol = 1e-3,
                                      SINKS = None,
                                      tNow = 0,
                                      fspErrorCondition = fspErrorCondition)
                                                
        if not tryAgain:
            break
        if (m > 300):
            print('Expokit expansion truncated at 300')
            w, err, hump, Time_array_out, P_array, P_lost, tryAgain, \
                te, ye = mexpv_modified_2(t = t, A = A, v = P0,
                                          tol = 1e-8, m = m,
                                          N_prt = None,
                                          Time_array = Time_array,
                                          fspTol = 1e-3,
                                          SINKS = None,
                                          tNow = 0,
                                          fspErrorCondition = fspErrorCondition)
            
        m += 5
    # while tryAgain ...
    
numArrayDimensions = 6
numSimulations = 5
arrayDimensions = np.floor(np.logspace(2, 3, numArrayDimensions))
simDataTemplateArray = np.ones((numSimulations, numArrayDimensions))

timeExpm = np.nan * simDataTemplateArray
timeExpokit = np.nan * simDataTemplateArray
timeODE23s = np.nan * simDataTemplateArray

diff_expm_expokit = np.nan * simDataTemplateArray
diff_expokit_ode23s = np.nan * simDataTemplateArray
diff_expm_ode23s = np.nan * simDataTemplateArray

startPt = 1 # Starting timepoint 
numPt = 2 # Total number of timepoints

COL_AXIS = 0

for index in range(len(arrayDimensions) - 1, -1, -1):
    N = int(arrayDimensions[index]);

    for simCntr in range(numSimulations):
        
        # Create an N-by-N random matrix. Determine the largest five elements
        # of each column, and use the smallest such element in each column to
        # create a threshold for the matrix. In this way, the overall matrix
        # will become very sparse, so we will then convert it to one. There are
        # various formats available within scipy.sparse, but for greatest
        # compatibility with sparsity-aware ODE solvers, we will choose either
        # the CSR or CSC format.
        
        rng = np.random.default_rng()
        A = rng.random((N, N))
        Athresh = np.apply_along_axis(get_threshold, COL_AXIS, A)
        Amasked = np.ma.masked_where(A < Athresh, A)
        A = Amasked.filled(fill_value = 0)
        A = csr_array(A) # csr_matrix(A)
        
        # Sum all elements in each column, place these on the diagonals of a
        # new matrix, and subtract the latter from A:
            
        sumA = A.sum(axis = COL_AXIS)
        diagFromSumA = diags([sumA], [0], (N, N), format = "csr")
        A = A - diagFromSumA
        
        # Create a random initial probability vector, normalizing it so that
        # its sum is one:
        
        P0 = rng.random((N, 1))
        P0 = P0 / P0.sum()
        
        # Solve the ODE via expm_multiply and record the time required in the 
        # appropriate matrix. (We refer to the operation as "expm," but scipy
        # provides an expm_multiply that includes the taking of the dot product
        # of e^A and P0.)
        
        timeExpm[simCntr][index] = timeit.timeit(
            stmt = """expm_multiply(A, P0, start = startPt,
            stop = startPt + numPt,
            num = numPt, endpoint = False)""", number = 1, globals = globals())

        # Solve the ODE via a modified version of expokit and record the time 
        # required in the appropriate matrix.
        
        timeExpokit[simCntr][index] = timeit.timeit(
            stmt = "test_mexpv(t = t, A = A, v = P0)", number = 1,
            globals = globals())
        
        # %% ODE Calculation
        # ode_opts = odeset('Jacobian', A, 'Vectorized','on','JPattern',A~=0,'relTol',1e-8, 'absTol', 1e-10);
        # rhs = @(t,x)A*x;
        # tic
        # [tExport, yout] =  ode15s(rhs, [0,t/2,t], P0);
        # timeODE23s(j,index) = toc;

        # diff_expm_expokit(j,index) = sum(abs(PfExpokit-expAt_P));
        # diff_expm_ode23s(j,index) = sum(abs(expAt_P-yout(end,:)'));
        # diff_expokit_ode23s(j,index) = sum(abs(PfExpokit-yout(end,:)'));

    # We will now generate figures comparing the running times and results of
    # the various ODE solvers tested.
    
    # Running times
    # figure(2)
    # subplot(2,1,1);
    meanTimeExpm = np.mean(timeExpm, axis = COL_AXIS)
    meanTimeExpokit = np.mean(timeExpokit, axis = COL_AXIS)
    np.savetxt('meanTimeExpm.txt', meanTimeExpm)
    np.savetxt('meanTimeExpokit.txt', meanTimeExpokit)
    np.savetxt('arrayDimensions.txt', arrayDimensions)
    plt.loglog(arrayDimensions, meanTimeExpm, '-s', label = 'expm')
    plt.loglog(arrayDimensions, meanTimeExpm, '-o', label = 'expokit')
    # loglog(arrayDimensions,mean(timeExpm),'-s',arrayDimensions,mean(timeExpokit),'-o',arrayDimensions,mean(timeODE23s),'-^');
    # legend('expm','expokit','ode23s')
    plt.xlabel('Number of states')
    plt.ylabel('Computational time')
    # for index in range(numArrayDimensions):
    #     plt.annotate(
    #         '%f sec for n = %d' % (meanTimeExpm[index], arrayDimensions[index]), xy=(index,index))
    plt.legend(loc = 'best')
    plt.savefig('foo.pdf')
    #set(gca,'fontsize',15)

    # # Numerical results
    # subplot(2,1,2);
    # plot(arrayDimensions,mean(diff_expm_expokit),'-o',arrayDimensions,mean(diff_expm_ode23s),'-s',arrayDimensions,mean(diff_expokit_ode23s),'-^')
    # legend('diff_expm_expokit','diff_expm_ode23s','diff_expokit_ode23s')

