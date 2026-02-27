import optuna
import os
from base.data import *
from algorithm import standard_GA

def objective(trial):
    '''
    This function is used in optuna optimization trials to find the 
    combination of hyperparameters that minimize a score that reflects
    both the fitness of the final elite (30%) as well as how often fitness
    improvements occured during the evolution (70%).
    '''
    # Suggested hyperparameters
    popsize = trial.suggest_int('popsize', 100, 1000)
    tournament_size = trial.suggest_int('tournament_size', 2, 20)
    crossover_prob = trial.suggest_float('crossover_prob', 0.5, 0.9)
    mutation_step = trial.suggest_int('mutation_step', 1, 500)
    update_ms = trial.suggest_float('update_ms', 1.001, 1.2)
    update_after_x_gens = trial.suggest_int('update_after_x_gens', 50, 200)
    pass_to_next_gen_free = trial.suggest_float('pass_to_next_gen_free', 0.0, 0.3)

    # Ensure log directory exists
    log_dir = 'logs_optuna'
    os.makedirs(log_dir, exist_ok=True)

    # Run the GA and return the full fitness curve over generations
    _, fitness_curve = standard_GA(
        data=data,
        popsize=popsize,
        tournament_size=tournament_size,
        crossover_prob=crossover_prob,
        mutation_step=mutation_step,
        update_ms=update_ms,
        update_ms_after_x_gens=update_after_x_gens,
        pass_to_next_gen_free=pass_to_next_gen_free,
        path_to_log=os.path.join(log_dir, f"log_optuna_trial_{trial.number}.txt"),
        seed= 42 + trial.number
    )

    # Compute final fitness and improvement frequency
    final_fitness = fitness_curve[-1]
    improvement_count = sum(
        1 for i in range(1, len(fitness_curve)) if fitness_curve[i] < fitness_curve[i - 1]
    )

    # Normalize both terms
    norm_fitness = final_fitness / 100       # assuming RMSE typically < 100
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

