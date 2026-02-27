import random
import numpy as np

CROSSOVER_PROB = 0.6
TOURNAMENT_SIZE = 3
PASS_TO_NEXT_GEN_FREE_PROB = 0.1

# TOURNAMENT SELECTION TO CHOOSE ONE PARENT FROM POPULATION
def tournament_selection(pop_dict, fit_dict, tournament_size = TOURNAMENT_SIZE):
    '''
    This function takes the pop_dict and fit_dict for a given
    generation as well as a tournament size and returns the 
    winner from the tournament as a list [id, genotype]. 
    NOTE: tournament selection consists of 2 steps 1. chose n 
    individuals from the population randomly (n =tournament size) 
    and 2. evaluate the fitnesses of the chosen individuals 
    and the best fitted wins.
    '''
    # Randomly select tournament_size individuals
    tournament_ids = random.sample(list(pop_dict.keys()), tournament_size)

    # Find the individual with the best (lowest) RMSE
    best_id = min(tournament_ids, key=lambda ind_id: fit_dict[ind_id])

    # Return the best individual
    return [best_id, pop_dict[best_id]]

# SELECTING WHAT OPERATION TO DO
def crossover_or_mut (crossover_prob = CROSSOVER_PROB, pass_to_next_gen_free = PASS_TO_NEXT_GEN_FREE_PROB):
    '''
    This function selects the operation to do based
    on a pre-established crossover probability as well as
    a probability that the chosen parent will pass to the
    next generation unchanged. The return is C (crossover),
    M (mutation) or P(pass to the next generation).
    '''
    # evaluate if the individual can go into the next generation unchanged
    pass_through = operator = random.uniform(0,1)
    if pass_through <= pass_to_next_gen_free:
        return ('P')

    # if it cannot, then we chose a reproduction operator
    operator = random.uniform(0,1) # chose random probability
    if operator <= crossover_prob:
        return ('C')
    else:
        return ('M')