import pandas as pd
import random
import json

from base.data import *
from base.individual import *
from base.population import *
from operators.crossovers import *
from operators.mutators import *
from operators.selectors import *
from operators.extinction import *

# hyperparameters that carry over from standard GA
FILENAME_MINMAX = "min_max_tuples.txt" 
POPSIZE = 768 
TOURNAMENT_SIZE = 19 
CROSSOVER_PROB = 0.83266628601947 
MUTATION_STEP = 464 
UPDATE_MS = 1.0907061943267877 
UPDATE_AFTER_X_GENS = 50 
PATH_TO_LOG = 'run_x_log.txt' # NOTE: THIS HAS TO BE CHANGED FOR EACH RUN!!!!! it's done automatically in the main loop as well as changing the seed 
PASS_TO_NEXT_GEN_FREE_PROB = 0.15660853661618737 
NGENS = 300 

# extinction hyperparameters that are tuned by Optuna
EXTINCTION_AVAILABLE_AFTER_GEN = 69
COOLDOWN = 42
GENS_OF_RELAXED_SELECTIVE_PRESSURE = 28
RELAXED_TOURNAMENT_SIZE = 8
EXTINCTION_SIZE_PCT = 0.2849567979136528

# extinction hyperparameters that are manually picked
ELITE_PCT_EXTINCTION_GEN = 0.25
DIVERSE_PCT_EXTINCTION_GEN = 0.5

def GA_with_extinction(data=data, filename=FILENAME_MINMAX, popsize=POPSIZE, n_gens=NGENS, tournament_size=TOURNAMENT_SIZE,
        crossover_prob=CROSSOVER_PROB, pass_to_next_gen_free=PASS_TO_NEXT_GEN_FREE_PROB, mutation_step=MUTATION_STEP, update_ms=UPDATE_MS,
        update_ms_after_x_gens=UPDATE_AFTER_X_GENS, extinction_available_after_gen=EXTINCTION_AVAILABLE_AFTER_GEN, extinction_size_pct=EXTINCTION_SIZE_PCT,
        elite_pct=ELITE_PCT_EXTINCTION_GEN, diverse_pct=DIVERSE_PCT_EXTINCTION_GEN, seed=42, cooldown = COOLDOWN, 
        relaxed_tournament_size = RELAXED_TOURNAMENT_SIZE, relaxed_selective_pressure_for_n_gens = GENS_OF_RELAXED_SELECTIVE_PRESSURE, 
        path_to_log=PATH_TO_LOG):
    '''
    This is the main function for the Genetic Algorithm with extinction events :)

    It enhances the standard GA with:
    - periodic extinction events triggered by elite fitness stagnation
    - reduced selective pressure for a limited number of generations after extinction

    It takes as input:
    - data: the training dataset to be used for fitness evaluation
    - filename: the file storing (min, max) ranges for each feature to generate individuals
    - popsize: the number of individuals per generation
    - n_gens: how many generations the GA will run (termination condition)
    - tournament_size: the default number of individuals sampled for tournament selection
    - crossover_prob: the probability of using crossover instead of mutation for reproduction
    - pass_to_next_gen_free: the probability that an individual passes unchanged to the next generation
    - mutation_step: the step size for geometric mutation (range of random changes)
    - update_ms: factor by which mutation_step is divided every generation (simulated annealing)
    - update_ms_after_x_gens: how many generations to wait before starting to update mutation_step
    - seed: random seed for reproducibility
    - path_to_log: path to the text file where the evolution run is logged (per generation), in the format:
        generation_number, elite_id, elite_genotype, elite_fitness, population_diversity

    - extinction_available_after_gen: number of generations to evolve before extinctions can be triggered
    - extinction_size_pct: what percentage of the population will be replaced in an extinction event
    - elite_pct: percentage (of the extinction generation size) reserved for top elite individuals
    - diverse_pct: percentage (of the extinction generation size) reserved for the most genetically diverse individuals
    NOTE: the remaining part of the extinction generation that is not elites or diverse individuals is occupied by
    randomly generated "immigrants" (generated the same way that the founder generation individuals are generated)
    - cooldown: number of generations to wait before another extinction can be triggered
    - relaxed_selective_pressure_for_n_gens: number of generations after extinction during which tournament size is reduced
    - relaxed_tournament_size: the smaller tournament size to use during relaxed selection periods

    Returns:
    - implicitly: a log text file with per-generation data for performance analysis and visualization
    - explicitly: the best individual of the final generation as a tuple: (elite_id, elite_genotype, elite_fitness)
    '''

    random.seed(seed)

    # create list of elite fitnesses for optuna optimization - THIS LIST IS NOT WIPED OUT, the values here are for optuna optimization
    fitness_curve = []
    
    # Create founder generation
    pop_dict = create_gen_zero(data, filename, popsize)
    pop_fit = evaluate_generation(data, pop_dict)
    pop_diversity = calculate_pop_diversity_variance(pop_dict)
    elite_id, elite_genotype, elite_phenotype = find_elite(pop_fit, pop_dict)
    fitness_curve.append(elite_phenotype)

    elite_fitness_history = [elite_phenotype] # create a list to track fitness of the elites to detect when stagnation happens - THIS LIST IS WIPED OUT AFTER EXTINCTION
    cooldown_counter = 0  # delay repeated extinctions
    selective_pressure_relaxation_counter = 0  # relaxed selection pressure after extinction

    print('Founder generation is done!')

    with open(path_to_log, 'w') as log_file:
        log_file.write("generation,best_id,best_genotype,best_fitness,diversity\n")
        log_file.write(f"0,{elite_id},\"{json.dumps(elite_genotype)}\",{elite_phenotype},{pop_diversity}\n")

    print('Starting evolution process!\n')

    # For a set number of generations, we evolve the population
    for gen in range(1, n_gens + 1):
        print('##############################################################################')
        print(f'Starting generation {gen}!\n')

        # Extinction trigger begins as false
        trigger_extinction = False

        # Extinction can only be triggered if there has not been one in a while  (controled by cooldown)
        ## and there's been a certain number of generations since the beggining of evolution (controled
        ### by extinction_available_after_gen)
        if cooldown_counter == 0 and gen >= extinction_available_after_gen:

            # if extinction is available to be triggered, we check for elite fitness stagnation
            if has_fitness_stagnated(elite_fitness_history):
                print("Extinction triggered due to elite fitness stagnation.")
                trigger_extinction = True
                cooldown_counter = cooldown # reset cooldown - there can only be another extinction in a few generations
                selective_pressure_relaxation_counter = relaxed_selective_pressure_for_n_gens # reduce pressure for some generations
        else:
            cooldown_counter = max(0, cooldown_counter - 1)

        # follow evolution either extinction generation or normal generation
        if trigger_extinction: # EXTINCTION GENERATION 
            # Calculate extinction generation size (as % of popsize)
            extinction_size = int(popsize * extinction_size_pct)

            # Calculate counts of each component type
            n_elites = int(extinction_size * elite_pct)
            n_diverse = int(extinction_size * diverse_pct)
            n_immigrants = extinction_size - n_elites - n_diverse

            print(f"Creating extinction generation of size {extinction_size}:")
            print(f"Elites: {n_elites}, Diverse: {n_diverse}, Immigrants: {n_immigrants}")

            # Build extinction population
            (pop_dict, pop_fit, pop_diversity) = making_extinction_generation(gen, pop_dict, pop_fit, n_elites=n_elites, m_moreDiverse=n_diverse,
                x_immigrants=n_immigrants, filename=filename)
            
            # Find elite of extinction generation
            elite_id, elite_genotype, elite_phenotype = find_elite(pop_fit, pop_dict)
            fitness_curve.append(elite_phenotype)

            # Clear elite fitness history after extinction
            elite_fitness_history = []

        else: # STANDARD GENERATION
            popnx_dict = {}
            counter_popnx_individuals = 0

            # Carry over elite
            elite_id_new = 'elite_' + str(gen)
            popnx_dict[elite_id_new] = elite_genotype
            print(f'{elite_id_new} (elite) is carried over unchanged to generation {gen}')

            while len(popnx_dict) < popsize:
                counter_popnx_individuals += 1
                operator = crossover_or_mut(crossover_prob, pass_to_next_gen_free)
                print(f'operator {operator} was chosen for individual {counter_popnx_individuals} of generation {gen}')

                # Adjust tournament size if selection pressure is relaxed
                tournament_size_this_gen = tournament_size
                if selective_pressure_relaxation_counter > 0:
                    tournament_size_this_gen = relaxed_tournament_size  # relaxed pressure
                    selective_pressure_relaxation_counter -= 1

                if operator == 'P':
                    lucky = tournament_selection(pop_dict, pop_fit, tournament_size_this_gen)
                    popnx_dict[lucky[0]] = lucky[1]
                    print(f'{lucky[0]} moves unchanged into generation {gen}\n')

                elif operator == 'M':
                    parent = tournament_selection(pop_dict, pop_fit, tournament_size_this_gen)
                    child = geometric_mutation(parent, mutation_step)
                    indi_id = 'indi_' + str(gen) + '_' + str(counter_popnx_individuals)
                    popnx_dict[indi_id] = child
                    print(f'{indi_id} is mutated and added into generation {gen}\n')

                elif operator == 'C':
                    parent1 = tournament_selection(pop_dict, pop_fit, tournament_size_this_gen)
                    parent2 = tournament_selection(pop_dict, pop_fit, tournament_size_this_gen)
                    child = geometric_crossover(parent1, parent2)
                    indi_id = 'indi_' + str(gen) + '_' + str(counter_popnx_individuals)
                    popnx_dict[indi_id] = child
                    print(f'{indi_id} is the result of crossover and added into generation {gen}\n')

            # Finalize new generation
            pop_dict = popnx_dict
            pop_fit = evaluate_generation(data, pop_dict)
            pop_diversity = calculate_pop_diversity_variance(pop_dict)
            elite_id, elite_genotype, elite_phenotype = find_elite(pop_fit, pop_dict)
            fitness_curve.append(elite_phenotype)

        # Update elite fitness list
        elite_fitness_history.append(elite_phenotype)

        if gen > update_ms_after_x_gens:
            mutation_step = mutation_step / update_ms

        with open(path_to_log, 'a') as log_file:
            log_file.write(f"{gen},{elite_id},\"{json.dumps(elite_genotype)}\",{elite_phenotype},{pop_diversity}\n")

        print(f'Generation {gen} is done!\n')

    return (find_elite(pop_fit, pop_dict), fitness_curve)