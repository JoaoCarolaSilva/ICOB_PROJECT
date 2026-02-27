from base.population import *
from base.individual import *
from base.data import *

MAX_DELTA = 10
FILENAME_MINMAX = "min_max_tuples.txt"

# EVALUATE IF FITNESS HAS STAGNATED
def has_fitness_stagnated(elite_fit_history, max_delta = MAX_DELTA):
    '''
    This function takes the history of fitnesses
    of the elites and evaluates whether the fitness of
    the elites has stagnated in the last 20 generations.
    If the difference between the max fitness of elites
    and the min fitness of elites in the last 20 generations
    is smaller than max_delta, then the fitness has stagnated. 
    This function returns true or false depending on 
    whether the fitness has stagnated or not.
    '''
    # get elite fitness values from last 20 generations
    elite_fit_history_last20 = elite_fit_history[-20:]

    # evaluate if the fitness has stagnated
    if max(elite_fit_history_last20) - min(elite_fit_history_last20) > max_delta:
        return (False)
    else:
        return (True)
    
# SELECTING SURVIVORS AND ADDING IMIGRANTS
def making_extinction_generation (gen, pop_dict, pop_fit, n_elites, m_moreDiverse, x_immigrants, filename = FILENAME_MINMAX):
    '''
    This function takes the current generation, pop_dict and pop_fit 
    and the numbers of elites, more diverse individuals and immigrants 
    that are going to make up the extinction generation and returns
    new post-extintion pop_dict and pop_fit as well as the filename of 
    the file needed to generate immigrants. NOTE: the returned pop_dict
    and pop_fit will be smaller than the ones for the generations that
    come previously and after. 
    '''
    # Preserve the original population before wiping to prevent errors
    original_pop_dict = pop_dict.copy()
    original_pop_fit = pop_fit.copy()

    # Start clean population and fitness dictionaries with different names 
    ## from the original population dictionaries
    pop_dict = {}
    pop_fit = {}

    elites = find_n_elites(original_pop_fit, original_pop_dict, n_elites) # get elites from original population
    mostDiverse = find_most_diverse(original_pop_dict, m_moreDiverse) # get most diverse individuals from original population
    immigrants = create_X_immigrants(data, x_immigrants, filename) # generate immigrants

    # ID counter for new individuals
    counter_popnx_individuals = 0

    # Add elites to the new population
    for elite in elites:
        counter_popnx_individuals += 1
        indi_id = f'indi_{gen}_{counter_popnx_individuals}'
        pop_dict[indi_id] = elite

    # Add most diverse individuals
    for indiv in mostDiverse:
        counter_popnx_individuals += 1
        indi_id = f'indi_{gen}_{counter_popnx_individuals}'
        pop_dict[indi_id] = indiv

    # Add new random immigrants
    for immigrant in immigrants:
        counter_popnx_individuals += 1
        indi_id = f'indi_{gen}_{counter_popnx_individuals}'
        pop_dict[indi_id] = immigrant

    # Evaluate fitness of new extinction generation
    pop_fit = evaluate_generation(data, pop_dict)

    # Calculate new diversity
    population_diversity = calculate_pop_diversity_variance(pop_dict)

    return (pop_dict, pop_fit, population_diversity)








