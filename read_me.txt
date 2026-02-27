This is the read me of the ICOB project of students Inês Wemans, João da Silva and Ricardo Reis :) in here, we mainly want to explain the structure of this project in terms of folders and files and where everything goes and what everything does.

Our project consists of 2 types of GAs: standard GAs and GAs with extinctions.

- GA: folder with all code regarding the standard GA

	-- base: code to get data and generate the base data structures that the GA will work on top of
		--- data.py: code for getting the training data
		--- individual.py: code for saving the values necessary for creating founder generation, generating individuals of the founding generation and evaluating fitness of one 			individual
		--- population.py: code for creating founder generation, evaluate fitness of the elements in   population, diversity of the population as a whole and finding the elite in 			the population

	-- operators: code for selection, geometric crossover and geometric mutation
		--- crossovers.py: code for geometric crossover
		--- mutators.py: code for geometric mutation
		--- selectors.py: code for selecting parents (via tournament selection) and selecting operator

	-- algorithm.py: the actual algorithm code, a good explanation of each hyperparameter can be found here

	-- min_max_tuples.txt: the file where we save intervals necessary to initialize founder generation at each run (DON'T WORRY IF IT'S NOT THERE, when the algorithm is run it is 			created if it does not already exist)

	-- parkinsons_CIFOBIO_TRAIN.csv: training data, 90% of the data given to us for this project

	-- main.py: this is what needs to be run to run and log algorithm performance ATTENTION!!!!! NEED TO ADJUST THE NUMBER OF RUNS, AS WELL AS THE NAMES/ DESTINATION FOLDERS OF THE 		PRODUCED LOGS, IF IT IS RUN AS IS, THE CURRENT FILES IN THE logs_with_best_params FOLDER WILL BE OVERWRITTEN

	-- optuna_trial.py: code for hyperparameter tuning using the Optuna library ATTENTION!!!!! NEED TO ADJUST THE NUMBER OF RUNS, AS WELL AS THE NAMES/ DESTINATION FOLDERS OF THE 			PRODUCED LOGS, IF IT IS RUN AS IS, THE CURRENT FILES IN THE logs_optuna FOLDER WILL BE OVERWRITTEN

	-- logs_with_best_params: 
		--- 10 runs of the algorithm with the best hyperparameters found by Optuna (created by running main.py, while adjusting the number of runs parameter as well as the file 			names)

	-- logs_optuna: 
		--- 50 logs of the Optuna trials, created and logged by running optuna_trial.py ATTENTION: THE HYPERPARAMETERS BEING OPTIMIZED HERE ARE THE EVOLUTION PARAMETERS

- GA_extinctions:
	-- base (same as in GA) except…
		--- population.py also contains code for finding a certain number of "weirdos" in a population, finding a certain number of elites in a population

	-- operators (same as in GA) except…
		---extinction.py: code to perform extinction and to evaluate stagtnation of lite fitness over generations

	-- algorithm.py: GA including implementation of extinctions (a good explanation of what each hyperparameter is can be found here)

	-- min_max_tuples.txt (in the case of extinction GAs this is also needed to generate immigrant individuals)

	-- parkinsons_CIFOBIO_TRAIN.csv (same as in GA)

	-- main.py: this is what needs to be run to run and log algorithm performance ATTENTION!!!!! NEED TO ADJUST THE NUMBER OF RUNS, AS WELL AS THE NAMES/ DESTINATION FOLDERS OF THE 		PRODUCED LOGS, IF IT IS RUN AS IS, THE CURRENT FILES IN THE logs_with_best_params FOLDER WILL BE OVERWRITTEN - since various repopulation strategies can be run depending on 		number of elites and "weirdos" in the extinction generation, which are manually picked, we recommend changing file names to be indicative of this

	-- optuna_trial.py: code for hyperparameter tuning using the Optuna library ATTENTION!!!!! NEED TO ADJUST THE NUMBER OF RUNS, AS WELL AS THE NAMES/ DESTINATION FOLDERS OF THE 			PRODUCED LOGS, IF IT IS RUN AS IS, THE CURRENT FILES IN THE logs_optuna FOLDER WILL BE OVERWRITTEN

	-- logs_with_best_params: 
		--- 10 runs of the algorithm with the best extinction hyperparameters found by Optuna EVOLUTION HYPERPARAMETERS WERE OPTIMIZED BEFOREHAND (created by running main.py, while 			adjusting the number of runs parameter as well as the file names)

	-- logs_optuna: 
		---50 logs of the Optuna trials, created and logged by running optuna_trial.py ATTENTION: THE HYPERPARAMETERS BEING OPTIMIZED HERE ARE THE EXTINCTION HYPERPARAMETERS, 			EVOLUTION HYPERPARAMETERS WERE OPTIMIZED BEFOREHAND 

- standardGA_alternative_hyperparameters: 
	-- logs of 2 alternative standard GAs, 10 runs each (GAs with different hyperparameters obtained using a random approach instead of Optuna)

- parkinsons_CIFOBIO_TEST.csv: test data (10% of the data provided for this project)

- parkinsons_CIFOBIO_TRAIN.csv: training data

- PERFORMANCE_ON_TEST_DATA.ipynb: script for evaluating performance of GAs on test data

- PERFORMANCE_ON_TRAINING_DATA.ipynb: script for evaluating performance of GAs on training data

- report.pdf: pdf of the report divided in introduction, methods, results and discussion, conclusion and references 



