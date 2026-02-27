import random
import os
import numpy as np
from base.data import *

FILENAME_MINMAX = "min_max_tuples.txt"


# GET MIN MAX FOR EACH FEATURE TO GENERATE THE INDIVIDUALS OF GEN ZERO
def get_data_min_max(data, filename = FILENAME_MINMAX):
    '''
    This function returns a list of tuples with length #features + 1,
    where each tuple is (min, max) for that feature. The first tuple
    is the interval for alpha0 so it is (min, max) of the whole data.
    If the file exists, it loads the list from the file. If not, it 
    computes it and writes it to the file. NOTE: this is done because,
    for each run, these intervals are necessary and computing them
    over and over is a waste of time.
    '''
    # If we already have a file with the intervals ...
    if os.path.exists(filename):
        with open(filename, "r") as f:
            list_of_tuples = [tuple(map(float, line.strip().split(","))) for line in f]
        return list_of_tuples

    # If we don't
    # Compute list of (min, max) tuples
    data_independent = data.iloc[:, :-1]
    list_of_tuples = []
    list_of_tuples.append((data_independent.min().min(), data_independent.max().max()))  # alpha0 range
    for feature in data_independent.columns:
        list_of_tuples.append((data_independent[feature].min(), data_independent[feature].max()))

    
    with open(filename, "w") as f:
        for t in list_of_tuples:
            f.write(f"{t[0]},{t[1]}\n")

    return list_of_tuples

# CREATE RANDOM INDIVIDUAL 
def create_individual(data, filename = FILENAME_MINMAX):
    '''this function takes the data, uses get_data_min_max
    funtion to calculate/get from file the intervals 
    from where the alpha coefficients will be generated 
    to make a random individual and returns an individual
    in the form of a list of alpha coefficients'''
    # get interval list
    list_of_tuples = get_data_min_max(data, filename = "min_max_tuples.txt")

    # initialize individual as a list
    individual = []

    # iterate over list of tuples to generate alpha coefficients to make up the individual
    for feature in list_of_tuples:
        alphan = random.uniform(feature[0], feature[1])
        individual.append(alphan)
    
    return(individual)

# FITNESS FUNCTION 
def calculate_fitness_RMSE(individual, data):
    '''
    This function calculates RMSE (measure of fitness) of an
    individual by receiving the individual (as list of coefficients
    where the 1st coefficient is the intercept) as well as the data
    '''
    # Split features and target
    X = data.iloc[:, :-1].values  # all columns except last
    y_true = data.iloc[:, -1].values  # last column

    # Separate intercept and coefficients
    intercept = individual[0]
    coefficients = np.array(individual[1:])

    # Compute predictions: y_pred = intercept + X @ coefficients
    y_pred = intercept + np.dot(X, coefficients)

    # Compute RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Return RMSE as fitness (lower is better)
    return rmse