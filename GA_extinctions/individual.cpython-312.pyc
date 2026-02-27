import os
import numpy as np
import pandas as pd
import random
import math as m

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

# GET N ELITES - JUST THE GENOTYPES ARE NEEDED
def find_n_elites(fit_dict, pop_dict, n):
    '''
    This function takes the fitness dictionary (pehotypes
    of the generation) and the population dictionary 
    (genotype of the generation) and returns genotype of 
    the top-n individuals with the lowest RMSE (best fitness). 
    '''
    # Sort by fitness (lower is better), then extract genotypes
    sorted_ids = sorted(fit_dict, key=fit_dict.get)[:n]

    return [pop_dict[id_] for id_ in sorted_ids]

# GET N MOST DISTANT INDIVIDUALS - JUST THE GENOTYPES ARE NEEDED
def euclidean_distance(a, b):
    '''
    This function performs euclidian distance between two
    individuals (individuals can be taken as points in an 
    n-dimentional space), taking the two individuals as
    input and returning the distance between them. NOTE: this
    is an auxiliary function needed to find most "different"
    genotypes in a generation.
    '''
    return m.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def find_most_diverse(pop_dict, m):
    '''
    This function returns the genotypes of the top-m most
    different individuals in a population in terms of genotype.
    If individuals are considered to be points in an n-dimentional
    space, the most different genomes are the individuals 
    that are more distant (eucledian distance)to all other 
    individuals in that n-dimentional space.
    '''
    ids = list(pop_dict.keys()) # get list of ids of individuals in population

    selected = [random.choice(ids)] # select a random individual/point
    ids.remove(selected[0]) # remove the randomly selected point from the id list

    # choose m most distant individuals from the current set of points
    while len(selected) < m:
        max_dist = -1 
        next_id = None

        # for each point not in set yet, compute its' distance to the
        ## points currently in the set
        for candidate in ids:
            min_dist = min(
                euclidean_distance(pop_dict[candidate], pop_dict[sid]) for sid in selected
            )

            # keep track of the point with the largest minimum distance
            ## from the points currenlty in the set
            if min_dist > max_dist:
                max_dist = min_dist
                next_id = candidate
        
        # the most distant point to the current set is added to the set
        selected.append(next_id)
        ids.remove(next_id)

    return [pop_dict[id_] for id_ in selected]

# CREATE X IMMIGRANTS
def create_X_immigrants(data, x_immigrants, filename = FILENAME_MINMAX):
    '''
    This function takes the data, the filename necessary for 
    generating random individuals whitin data limits and the
    number of immigrants to generate, in the same way that 
    the individuals for the founder generation are generated.
    '''
    x_immigrants_list = []

    for n in range(x_immigrants):
        individual = create_individual(data, filename)
        x_immigrants_list.append(individual)
    
    return (x_immigrants_list)