import optuna
import os
from base.data import *
from algorithm import GA_with_extinction

# hyperparameters that carry over from standard GA 
FILENAME_MINMAX = "min_max_tuples.txt"
POPSIZE = 549
TOURNAMENT_SIZE = 15
CROSSOVER_PROB = 0.788702092924693
MUTATION_STEP = 403
UPDATE_MS = 1.1720681938436632
UPDATE_AFTER_X_GENS = 167
PATH_TO_LOG = 'run_x_log.txt' # NOTE: THIS HAS TO BE CHANGED FOR EACH RUN!!!!!
PASS_TO_NEXT_GEN_FREE_PROB = 0.28278314693045764
NGENS = 300


def objective(trial):
    '''
    This function is used in optuna optimization trials to find the 
    combination of hyperparameters that minimize a score that reflects
    both the fitness of the final elite (30%) as well as how often fitness
    improvements occured during the evolution (70%). In this case we
    are only optimizing extinction hyperparameters as evolution hyperparameters
    have been optimized previously and are NOT optimizing the percentage
    of immigrants, elites and weirdoes in the extinction generation, we
    are doing egalitarian repopulation. NOTE: we use the same evolution
    hyperparameters across all 5 GAs with and without extinctions to
    be able to compare them better.
    '''

    # Tunable extinction hyperparameters
    extinction_available_after_gen = trial.suggest_int('extinction_available_after_gen', 10, 100)
    extinction_size_pct = trial.suggest_float('extinction_size_pct', 0.05, 0.3)
    cooldown = trial.suggest_int('cooldown', 10, 100)
    relaxed_tournament_size = trial.suggest_int('relaxed_tournament_size', 2, 10)
    relaxed_selective_pressure_for_n_gens = trial.suggest_int('relaxed_selective_pressure_for_n_gens', 5, 50)

    # DON'T WANT TO TUNE PERCENTAGES OF ELITES, IMMIGRANTS AND WEIRDOES, 
    # WANT TO DO TUNING CONSIDERING EGALITARIAN REPOPULATION
    elite_pct = 0.33
    diverse_pct = 0.33


    # Ensure log directory exists
    log_dir = 'logs_optuna'
    os.makedirs(log_dir, exist_ok=True)

    # Run the GA and return the full fitness curve over generations
    _, fitness_curve = GA_with_extinction(data=data, filename=FILENAME_MINMAX, popsize=POPSIZE, n_gens=NGENS, tournament_size=TOURNAMENT_SIZE,
        crossover_prob=CROSSOVER_PROB, pass_to_next_gen_free=PASS_TO_NEXT_GEN_FREE_PROB, mutation_step=MUTATION_STEP, update_ms=UPDATE_MS,
        update_ms_after_x_gens=UPDATE_AFTER_X_GENS, 
        
        extinction_available_after_gen=extinction_available_after_gen, extinction_size_pct=extinction_size_pct,
        elite_pct=elite_pct, diverse_pct=diverse_pct, seed=42 + trial.number, cooldown = cooldown, 
        relaxed_tournament_size = relaxed_tournament_size, relaxed_selective_pressure_for_n_gens = relaxed_selective_pressure_for_n_gens, 
        path_to_log=os.path.join(log_dir, f"log_optuna_trial_{trial.number}.txt")) 

    # Compute final fitness and improvement frequency
    final_fitness = fitness_curve[-1]
    improvement_count = sum(
        1 for i in range(1, len(fitness_curve)) if fitness_curve[i] < fitness_curve[i - 1]
    )

    # Normalize both terms
    norm_fitness = final_fitness / 100  # assuming RMSE typically < 100
    norm_improvements = improvement_count / len(fitness_curve)  # normalize to [0,1]

    # Combined score (lower is better for Optuna)
    alpha = 0.3  # weight for final fitness
    beta = 0.7   # weight for improvement frequency

    score = (alpha * norm_fitness) - (beta * norm_improvements)
    return score  

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best trial:")
print(study.best_trial)
