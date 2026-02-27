import random 

MUTATION_STEP = 10

def geometric_mutation (parent, mutation_step):
    '''
    This function takes a parent [x1, x2, x3, ..., xn] and the 
    mutation step and produces a child [x1+R1, x2+R2, x3+R3, ...,
    xn+Rn] where R1, R2, R3, ..., Rn are randomly generated
    numbers between [-ms, ms]
    '''
    parent_genotype = parent[1]  # extract genotype from parent
    child = [] # intialize child as list

    for feature in parent_genotype:
        R = random.uniform(-mutation_step, mutation_step)
        mutated_feature = feature + R
        child.append(mutated_feature)

    return child






