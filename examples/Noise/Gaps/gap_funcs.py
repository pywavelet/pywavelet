# Initial values plus sampling properties


from scipy.signal.windows import tukey
import numpy as np

ONE_HOUR = 60*60
def gap_routine(t, start_window, end_window, lobe_length = 3, delta_t = 10):

    start_window *= ONE_HOUR          # Define start of gap
    end_window *= ONE_HOUR          # Define end of gap
    lobe_length *= ONE_HOUR          # Define length of cosine lobes

    window_length = int(np.ceil(((end_window+lobe_length) - 
                                (start_window - lobe_length))/delta_t))  # Construct of length of window 
                                                                        # throughout the gap
    alpha_gaps = 2*lobe_length/(delta_t*window_length)      # Construct alpha (windowing parameter)
                                                    # so that we window BEFORE the gap takes place.
    
    window = tukey(window_length,alpha_gaps)   # Construct window

    new_window = []  # Initialise with empty vector
    j=0  
    for i in range(0,len(t)):   # loop index i through length of t
        if t[i] > (start_window - lobe_length) and (t[i] < end_window + lobe_length):  # if t within gap segment
            new_window.append(1 - window[j])  # add windowing function to vector of ones.
            j+=1  # incremement 
        else:                   # if t not within the gap segment
            new_window.append(1)  # Just add a one.
            j=0
        

    alpha_full = 0.1
    total_window = tukey(len(new_window), alpha = alpha_full)

    new_window *= total_window
    return new_window

from numba import njit, prange
@njit()
def get_Cov(Cov_Matrix, w_fft, w_fft_star, delta_f, PSD):
    """
        Compute the covariance matrix for a given set of parameters.

        Parameters
        ----------
        Cov_Matrix : numpy.ndarray (Complex128)
            The initial covariance matrix to be updated.
        w_fft : numpy.ndarray
            Two-sided fourier transform of window function.
        w_fft_star : numpy.ndarray
            An array representing the conjugate transpose of the weights.
        delta_f : float
            The frequency increment.
        PSD : numpy.ndarray
            The Power Spectral Density array.

        Returns
        -------
        numpy.ndarray
            The updated covariance matrix.

        Notes
        -----
        This function performs the following steps:
        1. Computes a prefactor using `delta_f`.
        2. Iterates over the upper triangle of the matrix to update its elements.
        3. Enforces Hermitian symmetry by adjusting the diagonal and adding the conjugate transpose.
    
        The function is optimized using Numba's `njit` and `prange` for parallel execution.
        """ 
    pre_fac = 0.5 * delta_f
    n_t = len(w_fft) # w_fft is two sided transform, so same dimensions as N
    for i in prange(0, n_t//2 +1 ): # For each column
        if i % 50 == 0:
            print(i)
        for j in range(i,n_t//2 + 1 ):  # For each row
             Cov_Matrix[i,j] = pre_fac  * np.sum(PSD * np.roll(w_fft, i)*np.roll(w_fft_star, j))
    
    diagonal_elements = np.diag(np.diag(Cov_Matrix)) # Take real part of matrix
    diagonal_elements_real = 0.5*np.real(diagonal_elements)
    Cov_Matrix = Cov_Matrix - diagonal_elements + diagonal_elements_real 
    Cov_Matrix = Cov_Matrix + np.conjugate(Cov_Matrix.T)
    return Cov_Matrix
