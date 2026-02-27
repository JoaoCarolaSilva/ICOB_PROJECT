import random 

def geometric_crossover(parent1, parent2):
    '''
    This function takes two parents [x1, x2, x3, ..., xn]
    and [y1, y2, y3, ..., yn] and produces a child by doing
    the following: [R1*x1 + (1-R1*y1), R2*x2 + (1-R2*y2),
    R3*x3 + (1-R3*y3), ..., Rn*xn + (1-Rn*yn)] where the R
    coefficients are random numbers generated between 0 and 1
    '''
    parent1_genotype = parent1[1] # get genotype of parent1
    parent2_genotype = parent2[1] # get genotype of parent2

    child = [] # initialize child as a list 

    for feature in range(len(parent2_genotype)):
        R = random.uniform(0, 1)
        gene = R * parent1_genotype[feature] + (1 - R) * parent2_genotype[feature]
        child.append(gene)

    return child
