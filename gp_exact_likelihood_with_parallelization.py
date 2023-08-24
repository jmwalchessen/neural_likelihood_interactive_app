import numpy as np
import sys
import scipy
from scipy import linalg
from numpy import linalg
import ray

def construct_norm_matrix(minX, maxX, minY, maxY, n):
    # create one-dimensional arrays for x and y
    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    X_matrix = (np.repeat(X, n**2, axis = 0)).reshape((n**2, n**2))
    Y_matrix = (np.repeat(Y, n**2, axis = 0)).reshape((n**2, n**2))
    longitude_squared = np.square(np.subtract(X_matrix, np.transpose(X_matrix)))
    latitude_squared = np.square(np.subtract(Y_matrix, np.transpose(Y_matrix)))
    norm_matrix = np.sqrt(np.add(longitude_squared, latitude_squared))
    return norm_matrix

def construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale):

    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    exp_kernel = variance*np.exp((-1/lengthscale)*norm_matrix)
    return(exp_kernel)

def construct_exp_kernel_without_variance_from_norm_matrix(norm_matrix, lengthscale):

    exp_kernel_without_variance = np.exp((-1/lengthscale)*norm_matrix)
    return(exp_kernel_without_variance)

def generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale, number_of_replicates, seed_value):

    kernel = construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale)
    np.random.seed(seed_value)
    z_matrix = np.random.multivariate_normal(np.zeros(n**2), np.identity(n**2), number_of_replicates)
    C = np.linalg.cholesky(kernel)
    y_matrix = np.matmul(np.transpose(C), np.transpose(z_matrix))

    return y_matrix

def compute_log_likelihood_fast_with_cholesky(y, C_without_variance, diag_C_without_cholesky, variance, n):

    C = np.sqrt(variance)*C_without_variance
    diagC = np.sqrt(variance)*diag_C_without_cholesky
    D = scipy.linalg.solve_triangular(C, y, lower = True)
    num = (-1/2)* np.matmul(np.transpose(D), D)
    denom = (n/2)*np.log(2*np.pi) + (1/2)*2*np.sum(np.log(diagC))
    log_likelihood = num - denom
    return log_likelihood

@ray.remote
def inner_for_loop_in_log_likelihood_field_with_cholesky(current_lengthscale, y, possible_variances, norm_matrix, n):

    likelihood_vector = np.zeros((len(possible_variances)))
    kernel_without_variance = construct_exp_kernel_without_variance_from_norm_matrix(norm_matrix, current_lengthscale)
    cholesky_without_variance = np.linalg.cholesky(kernel_without_variance)
    diag_cholesky_without_variance = np.diag(cholesky_without_variance)

    for i in range(0, len(possible_variances)):
        current_variance = possible_variances[i]
        likelihood_vector[i] = compute_log_likelihood_fast_with_cholesky(y, cholesky_without_variance, diag_cholesky_without_variance, current_variance, n)
        
    return likelihood_vector

def produce_parallelized_ll_field(operation, inputs):
    return ray.get([operation.remote(input[0], input[1], input[2], input[3], input[4]) for input in
                    inputs])

def execute_parallelized_ll_field(y, possible_variances, possible_lengthscales, minX, maxX, minY, maxY, n):

    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    ray.init(num_cpus = 16)
    inputs = [(current_lengthscale, y, possible_variances, norm_matrix, n) for current_lengthscale
              in possible_lengthscales]
    output = produce_parallelized_ll_field(inner_for_loop_in_log_likelihood_field_with_cholesky, inputs)
    ray.shutdown()
    output = (np.asarray(output))
    return output


