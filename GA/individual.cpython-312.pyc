import os
import numpy as np
import pandas as pd
import random

from base.data import *
from base.individual import *


FILENAME_MINMAX = "min_max_tuples.txt"
POPSIZE = 35

# CREATE GEN ZERO FULL OF RANDOMLY GENERATED INDIVIDUALS
def create_gen_zero(data, filename = FILENAME_MINMAX, popsize = POPSIZE):
    '''
    This function takes the data, the filename necessary for 
    generating random individuals whitin data limits and the
    popsize to know how many random individuals to generate.
    The generation is saved in a dictionary where keys are 
    individual id's (indi_x where x is a number) and values
    are the individuals' genotypes (as a list of coefficients)
    '''
    pop_dict = {} # initialize population dictionary

    for n in range(popsize):
        individual = create_individual(data, filename)
        key = 'indi_0_' + str(n)
        pop_dict[key] = individual
    
    return (pop_dict)

# EVALUATE FITNESS OF THE GENERATION
def evaluate_generation(data, pop_dict):
    '''
    This function takes the data, the dictionary with all
    individuals in a population (keys are individual id
    indi_gen_ind) and returns a dictionary with fitnesses
    of the individuals in that populations where the keys 
    are the same as the keys in the pop_dict and values are
    the corresponding fitness values calculated by RMSE.
    '''
    fit_dict = {} #initialize fitness dictionary for the generation

    for id, indi in pop_dict.items():
        # evaluate fitness of individual
        fitness_rmse = calculate_fitness_RMSE(indi, data)
        fit_dict[id] = fitness_rmse
    
    return (fit_dict)

# FIND ELITE - THE INDIVIDUAL WITH BEST FITNESS
def find_elite(fit_dict, pop_dict):
    '''
    This function takes the fitness dictionary (phenotype)
    and the population dictionary (genotype) and returns
    the best individual in formatted as a list: 
    [individual id, individual genotype, individual pheotype]
    '''
    # Find the individual ID with the minimum RMSE (best fitness)
    best_id = min(fit_dict, key=fit_dict.get)

    # Get genotype and phenotype
    best_genotype = pop_dict[best_id]
    best_phenotype = fit_dict[best_id]

    # Return as a list
    return ([best_id, best_genotype, best_phenotype])

# CALCULATE POPULATION DIVERSITY
def calculate_pop_diversity_variance(pop_dict):
    '''
    This function takes the pop_dict where keys
    are individual ids and values are individual
    genotypes and returns the variance value that
    will serve as a measure of diversity in the 
    population
    '''
    # Extract all genotypes as a 2D NumPy array
    genotype_matrix = np.array(list(pop_dict.values()))

    # Calculate variance across each column (i.e., each gene/weight position)
    gene_variances = np.var(genotype_matrix, axis=0)

    # Sum all variances to get a single diversity value
    total_variance = np.sum(gene_variances)

    return total_variance
