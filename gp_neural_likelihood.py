import tensorflow
from tensorflow import keras
from keras.models import model_from_json
import pickle
import numpy as np
import sklearn
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as patch_polygon

image_size = 25 
image_name = str(image_size) + "_by_" + str(image_size)
version = "final_version"

json_file_name = ("gaussian_process/nn/models/" + image_name + "/" + version
+ "/model/" + "gp_" + image_name + "_" + version + "_nn.json")

weights_file_name = ("gaussian_process/nn/models/" + image_name + "/" + version
+ "/model/" + "gp_" + image_name + "_" + version + "_nn_weights.h5")

json_file = open(json_file_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
parameter_classifier = model_from_json(loaded_model_json)
parameter_classifier.load_weights(weights_file_name)

with open(("gaussian_process/nn/calibration/model/" + image_name + "/" + version + "/"
           + "logistic_regression_model_with_logit_transformation.pkl"), 'rb') as f:
    clf = pickle.load(f) 

possible_lengthscales = [.05*i for i in range(1,41)]
possible_sigmas = [.05*i for i in range(1,41)]

def multi_psi(images, parameters):

    classifier_outputs = parameter_classifier.predict([images, parameters])
    psi_values = np.zeros(shape = (classifier_outputs.shape[0], 1))
    for i in range(classifier_outputs.shape[0]):
        output = classifier_outputs[i,:]
        if(output[1] == 1):
            psi_value = (1-output[0])/output[0]
        else:
            psi_value = output[1]/(1-output[1])

        psi_values[i,:] = psi_value

    return psi_values

def multi_calibrated_psi(images, parameters):

    predictions = parameter_classifier.predict([images, parameters])
    z_scores = ((np.log(predictions/(1-predictions)))[:,0]).reshape((-1,1))
    z_scores[z_scores == np.inf] = np.amax(z_scores[z_scores != np.inf])
    z_scores[z_scores == np.NaN] = np.amax(z_scores[z_scores != np.inf])
    z_scores[z_scores == -1*np.inf] = np.amin(z_scores[z_scores != -1*np.inf])
    classifier_outputs = (1 - clf.predict_proba(z_scores))
    psi_values = np.zeros(shape = (classifier_outputs.shape[0], 1))
    for i in range(classifier_outputs.shape[0]):
        output = classifier_outputs[i,:]
        if(output[1] == 1):
            psi_value = (1-output[0])/output[0]
        else:
            psi_value = output[1]/(1-output[1])

        psi_values[i,:] = psi_value

    return psi_values

def produce_psi_field(possible_ranges, possible_smooths, image, n):

    number_of_parameter_pairs = len(possible_ranges)*len(possible_smooths)
    image = image.reshape((1, n, n, 1))
    image_matrix = np.repeat(image, number_of_parameter_pairs, axis  = 0)
    ranges = (np.repeat(np.asarray(possible_ranges), len(possible_smooths), axis = 0)).reshape((number_of_parameter_pairs, 1))
    smooths = []
    smooths = (np.array(sum([(smooths + possible_smooths) for i in range(0, len(possible_ranges))], []))).reshape((number_of_parameter_pairs,1))
    parameter_matrix = np.concatenate([ranges, smooths], axis = 1)
    psi_field = (multi_psi(image_matrix, parameter_matrix)).reshape((len(possible_smooths), len(possible_ranges)))

    return psi_field

def produce_calibrated_psi_field(possible_ranges, possible_smooths, image, n):

    number_of_parameter_pairs = len(possible_ranges)*len(possible_smooths)
    image = image.reshape((1, n, n, 1))
    image_matrix = np.repeat(image, number_of_parameter_pairs, axis  = 0)
    ranges = (np.repeat(np.asarray(possible_ranges), len(possible_smooths), axis = 0)).reshape((number_of_parameter_pairs, 1))
    smooths = []
    smooths = (np.array(sum([(smooths + possible_smooths) for i in range(0, len(possible_ranges))], []))).reshape((number_of_parameter_pairs,1))
    parameter_matrix = np.concatenate([ranges, smooths], axis = 1)
    psi_field = (multi_calibrated_psi(image_matrix, parameter_matrix)).reshape((len(possible_smooths), len(possible_ranges)))

    return psi_field

def produce_neural_likelihood_confidence_region(neural_likelihood_surface, possible_lengthscales, possible_variances, C):

    max_field_value = np.log(np.max(neural_likelihood_surface))
    field_difference = 2*(max_field_value - np.log(neural_likelihood_surface))
    confidence_grid = np.where(field_difference <= C, 1, 0)

    variance_values = []
    lengthscale_values = []
    
    for i in range(0, confidence_grid.shape[0]):
        if(np.array(np.where((confidence_grid[i,:]) == 1)).any()):
            #min_val = (np.array(np.where((confidence_grid[i,:]) == 1))).min()
            max_val = (np.array(np.where((confidence_grid[i,:]) == 1))).max()
            variance_values.append(possible_variances[i])
            lengthscale_values.append(possible_lengthscales[max_val])

    for i in range((confidence_grid.shape[0] - 1), 0, -1):
        if(np.array(np.where((confidence_grid[i,:]) == 1)).any()):
            min_val = (np.array(np.where((confidence_grid[i,:]) == 1))).min()
            variance_values.append(possible_variances[i])
            lengthscale_values.append(possible_lengthscales[min_val])

    confidence_region = np.zeros((len(variance_values),2))
    confidence_region[:,0] = lengthscale_values
    confidence_region[:,1] = variance_values

    return confidence_region

#This function produces the 95 percent approximate confidence region over the parameter grid for a given exact log likelihood surface
    #parameters:
        #exact_likelihood_surface: 40 by 40 matrix, exact log likelihood surface for a given realization of the GP
        #possible_lengthscales: values of lengthscales on the parameter grid
        #possible_variances: values of variances on the parameter grid
        #C: cut off value that corresponds to 95 percent coverage for chi-distribution with 2 degrees of freedom (dimension of parameter space)
def produce_exact_likelihood_confidence_region(exact_likelihood_surface, possible_lengthscales,
                                               possible_variances, C):

    max_field_value = np.max(exact_likelihood_surface)
    field_difference = 2*(max_field_value - exact_likelihood_surface)
    confidence_grid = np.where(field_difference <= C, 1, 0)
    
    #if we assume convex surface

    variance_values = []
    lengthscale_values = []
    
    for i in range(0, confidence_grid.shape[0]):
        if(np.array(np.where((confidence_grid[i,:]) == 1)).any()):
            #min_val = (np.array(np.where((confidence_grid[i,:]) == 1))).min()
            max_val = (np.array(np.where((confidence_grid[i,:]) == 1))).max()
            variance_values.append(possible_variances[i])
            lengthscale_values.append(possible_lengthscales[max_val])

    for i in range((confidence_grid.shape[0] - 1), 0, -1):
        if(np.array(np.where((confidence_grid[i,:]) == 1)).any()):
            min_val = (np.array(np.where((confidence_grid[i,:]) == 1))).min()
            variance_values.append(possible_variances[i])
            lengthscale_values.append(possible_lengthscales[min_val])

    confidence_region = np.zeros((len(variance_values),2))
    confidence_region[:,0] = lengthscale_values
    confidence_region[:,1] = variance_values

    return confidence_region