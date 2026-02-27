import pandas as pd
import random
import json


from base.data import *
from base.individual import *
from base.population import *
from operators.crossovers import *
from operators.mutators import *
from operators.selectors import *
from algorithm import *

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

# SET UP LOGS FOR A SET NUMBER OF RUNS
NRUNS = 10
log_file_prefix = 'weirdo_extinction_GA_run'
log_file_suffix = '_log.txt'
log_dir = 'logs_with_best_params'

# Create the log folder directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Generate full paths to log files
paths_to_log = [
    os.path.join(log_dir, f"{log_file_prefix}{n}{log_file_suffix}")
    for n in range(1, NRUNS + 1)
]
                
# EXECUTING AND LOGGING NRUNS
for run in range(len(paths_to_log)):
    print()
    print()
    print()
    print(f'RUN {run}')
    print()
    print()
    print()
    # Executions happens here
    GA_with_extinction(data=data, filename=FILENAME_MINMAX, popsize=POPSIZE, n_gens=NGENS, tournament_size=TOURNAMENT_SIZE,
        crossover_prob=CROSSOVER_PROB, pass_to_next_gen_free=PASS_TO_NEXT_GEN_FREE_PROB, mutation_step=MUTATION_STEP, update_ms=UPDATE_MS,
        update_ms_after_x_gens=UPDATE_AFTER_X_GENS,extinction_available_after_gen=EXTINCTION_AVAILABLE_AFTER_GEN, extinction_size_pct=EXTINCTION_SIZE_PCT,
        elite_pct=ELITE_PCT_EXTINCTION_GEN, diverse_pct=DIVERSE_PCT_EXTINCTION_GEN, cooldown=COOLDOWN,
        relaxed_selective_pressure_for_n_gens=GENS_OF_RELAXED_SELECTIVE_PRESSURE, relaxed_tournament_size=RELAXED_TOURNAMENT_SIZE,
        seed=42 + run, path_to_log=paths_to_log[run]
    )


