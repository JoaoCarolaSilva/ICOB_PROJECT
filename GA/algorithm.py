import pandas as pd
import random
import json


from base.data import *
from base.individual import *
from base.population import *
from operators.crossovers import *
from operators.mutators import *
from operators.selectors import *

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


def standard_GA (data = data, filename = FILENAME_MINMAX, popsize = POPSIZE, n_gens = NGENS,
        tournament_size = TOURNAMENT_SIZE, crossover_prob = CROSSOVER_PROB, 
        pass_to_next_gen_free = PASS_TO_NEXT_GEN_FREE_PROB, mutation_step = MUTATION_STEP, 
        update_ms = UPDATE_MS, update_ms_after_x_gens = UPDATE_AFTER_X_GENS ,seed = 42, 
        path_to_log = PATH_TO_LOG):
    '''
    This is the main function for the standard GA :) 

    It takes as input:
    -the data

    -the filename of the file with the intervals needed to generate generation zero for each run

    -the intended population size

    -the termination condition - how many generations are we going to evolve for 

    -tournament size for selection of parents for reproduction

    -the crossover probability (used to choose operator, assuming that mutation probability
    is 1-crossover probability) - INDIVIDUALS THAT REPRODUCE

    -the pass to next generation probability which is the probability that a parent chosen by tournament
    selection passes unchanged to the next generation - INDIVIDUALS THAT DON'T REPRODUCE

    -mutation step for geometric mutation to know the range of the change from parent to mutated child

    -update_ms parameter that serves to adjust mutation step as generations go by in a simulated annealing 
    inspired way (earlier generations have more drastic mutations and later generations suffer more delicate 
    adjustments by way of mutation) - at each generation mutation_step = mutation_step / update_ms

    -update_ms_after_x_gens parameter that serves to control when the ms parameter starts being updated,
    it only starts being updated after x generations

    -seed for reproducibility

    -path to a txt file where the run will be logged per generation, each line will represent a generation in
    the following way: gen_number, best individual (id and genotype), fitness of the best individual and 
    diversity in that generation (measured by variance)


    And it returns: 
    -(implicitly) a log txt file with information necessary for performance analysis for the run(plots, statistics)
    -the best individual of the last generation and its' fitness
    '''
    # set seed
    random.seed(seed)

    # create list of elite fitnesses for optuna optimization
    fitness_curve = []
    
    # Generate generation zero and calculate its' fitness and diversity
    pop_dict = create_gen_zero(data, filename, popsize)
    pop_fit = evaluate_generation(data, pop_dict)
    pop_diversity = calculate_pop_diversity_variance(pop_dict)
    elite_id = find_elite(pop_fit, pop_dict)[0]
    elite_genotype = find_elite(pop_fit, pop_dict)[1]
    elite_phenotype = find_elite(pop_fit, pop_dict)[2]
    fitness_curve.append(elite_phenotype)

    # PRINT FOUNDER GEN IS DONE
    print('Founder generation is done!')

    # Log founder generation into log txt
    with open(path_to_log, 'w') as log_file:
        log_file.write("generation,best_id,best_genotype,best_fitness,diversity\n")
        log_file.write(f"0,{elite_id},\"{json.dumps(elite_genotype)}\",{elite_phenotype},{pop_diversity}\n")

    # PRINT STARTING EVOLUTION
    print('Starting evolution process!')
    print()

    # Repeat for n generations
    for gen in range (1, n_gens+1):

        # PRINT GEN N
        print('##############################################################################')
        print(f'Starting generation {gen}!')
        print()
        
        # Generate dictionary to keep genomes of next generation
        popnx_dict = {}
        counter_popnx_individuals = 0  # Start from 0 since we add elite first

        # Add elite individual to next generation
        elite_id_new = 'elite_' + str(gen)
        popnx_dict[elite_id_new] = elite_genotype
        print(f'{elite_id_new} (elite) is carried over unchanged to generation {gen}')

        # Fill up next generation by i) reproduction M or C or ii) passing to next generation unchanged
        while len(popnx_dict.keys()) < popsize:

            counter_popnx_individuals += 1 # move to next individual, next id

            # Choose operator (we do this before picking the parent bc depending 
            ## on operation we choose a different number of parents) 
            operator = crossover_or_mut (crossover_prob, pass_to_next_gen_free)

            # PRINT CHOSEN OPERATOR FOR INDIVIDUAL N
            print(f'operator {operator} was chosen for individual {counter_popnx_individuals} of generation {gen}')


            # The individual passes to the next generation unchanged
            if operator == 'P':

                # Chose one individual to pass into the next generation unchanged
                lucky_bastard = tournament_selection(pop_dict, pop_fit, tournament_size)

                # Add them to the next gen
                indi_id = lucky_bastard[0] # get id to be key
                indi_genotype = lucky_bastard[1] # get genotype to be value

                popnx_dict[indi_id] = indi_genotype

                # PRINT 
                print(f'{indi_id} moves unchanged into generation {gen}')
                print()
            
            # The individual suffers mutation before passing to next generation
            elif operator == 'M':

                # Chose one individual to be mutated
                parent = tournament_selection(pop_dict, pop_fit, tournament_size)

                # Mutate the individual
                child = geometric_mutation (parent, mutation_step)

                # Get id for individual
                indi_id= 'indi_' + str(gen) + '_' + str(counter_popnx_individuals)

                # Add individual to next gen
                popnx_dict[indi_id] = child

                # PRINT 
                print(f'{indi_id} is mutated and added into generation {gen}')
                print()

            # Two individuals do crossover 
            elif operator == 'C':

                # Choose two parents
                parent1 = tournament_selection(pop_dict, pop_fit, tournament_size)
                parent2 = tournament_selection(pop_dict, pop_fit, tournament_size)

                # Do crossover
                child = geometric_crossover(parent1, parent2)

                # Get id for individual
                indi_id= 'indi_' + str(gen) + '_' + str(counter_popnx_individuals)

                # Add individual to next gen
                popnx_dict[indi_id] = child

                # PRINT 
                print(f'{indi_id} is the result of crossover and added into generation {gen}')
                print()

        # New generation is done, replace the last generation
        pop_dict = popnx_dict

        # Calculate fitness, diversity of next generation as well as its' elite
        pop_fit = evaluate_generation(data, pop_dict)
        pop_diversity = calculate_pop_diversity_variance(pop_dict)
        elite_id = find_elite(pop_fit, pop_dict)[0]
        elite_genotype = find_elite(pop_fit, pop_dict)[1]
        elite_phenotype = find_elite(pop_fit, pop_dict)[2]
        fitness_curve.append(elite_phenotype)

        # Update mutation step after each generation if we have passed x generations
        if gen > update_ms_after_x_gens:
            mutation_step = mutation_step/update_ms

        # Log generation into run log
        with open(path_to_log, 'a') as log_file:
            log_file.write(f"{gen},{elite_id},\"{json.dumps(elite_genotype)}\",{elite_phenotype},{pop_diversity}\n")

        # PRINT GEN n IS DONE
        print(f'Generation {gen} is done!')
        print()
    
    # Return the last elite - id, genotype, fitness and the fitness curve
    return (find_elite(pop_fit, pop_dict), fitness_curve)