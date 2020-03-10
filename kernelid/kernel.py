import numpy as np


def fir_regressor(u, p, d=0, y=None):
    N = u.shape[0]

    PHI = np.zeros((N-p-d+1, p))
    for row_idx in range(N-p-d+1):
        idx_first_reg = row_idx+p-1
        idx_last_reg = row_idx-1

        PHI[row_idx, :] = u[idx_first_reg :(None if idx_last_reg == -1 else idx_last_reg):-1].ravel()

    if y is not None:
        Y = y[p+d-1:]
        return PHI, Y
    else:
        return PHI

def  DC_kernel(i, j, c, alpha, beta):
    """ Computes the Diagonal/Correlated kernel element in position (i,j).

    It is based on Formula (14) of "The Effect of Prior Knowledge on the Least Costly Identification Experiment"
    By G. Birpoutsoukis and Xavier Bombois (Similar to "Kernel methods in system identification, machine learning and
    function estimation: A survey" By G. Pillonetto et al.)
    """

    # Diagonal / correlated kernel
    ker_val = c*np.exp(-alpha*abs(i-j)) * np.exp(-beta*(i+j)/2)
    return(ker_val)


def kernel_covariance(kernel_fun, p):
    jj, ii = np.meshgrid(np.arange(p), np.arange(p))
    K = kernel_fun(ii, jj)
    return K


def  get_marglik(Y, K):
    """ Compute the marginal likelihood:

    Parameters:
    -----------
    """
    L = np.linalg.cholesky(K)
    U =  np.linalg.solve(L, np.eye(L.shape[0]))
    Kinv = np.linalg.solve(L.T, U)
    marglik = Y.T @ Kinv @Y
    marglik += 2 * np.sum(np.log(np.diag(L))) # Gaussian Processes for Machine Learning, Rasmussen and Williams, A.18
    return marglik

if __name__ == '__main__':
    i = 10
    j = 20

    c = 5.6
    alpha = 0.033
    beta = 0.232

    p = 10

    kernel_fun = lambda i, j: DC_kernel(i, j, c, alpha, beta)
    jj, ii = np.meshgrid(np.arange(p), np.arange(p))
    K = DC_kernel(ii, jj, c, alpha, beta)

    K1 = kernel_covariance(kernel_fun, p)