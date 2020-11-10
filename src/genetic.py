from deap import tools
from deap import base
from deap import creator
#for graphic
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors

from src.individual import *
from src.population import *
from src.transfer import *

def genetic(src_struct, target, pred_inst, train_pos_target, 
            train_neg_target, train_facts_target, test_pos_target, 
            test_neg_target, test_facts_target, NUM_GEN=600, 
            pop_size=10, crossover=0.6, mutation=0.3):

    # NUM_GEN = 600

    pop = Population(pop_size)
    best_evaluates = []

    
    pop.construct_population(src_struct, target, pred_inst)
    # print(f"First Population with size {str(pop.pop_size)}:")
    # pop.print_pop()

    pop.evaluation(pop.population, train_pos_target, 
                   train_neg_target, train_facts_target,
                   test_pos_target, test_neg_target, 
                   test_facts_target)

    # pop.hof.update(pop.population)

    # return pop.population
    for generation in range(NUM_GEN):
        pred_inst.bk_source = pred_inst.bk_target
        # print(pred_inst.bk_source)
        for ind in pop.population:
            ind.source_tree = ind.individual_trees
            ind.predicate_inst.bk_source = ind.predicate_inst.bk_target
        # print(pop.population[0].predicate_inst.bk_source)
            # ind.transfer.predicate_inst = ind.predicate_inst
        # best_individuals = pop.toolbox.selBest(pop.population, 1)
        # print(best_individuals, best_individuals[0].fitness.values)

        best_evaluates.append(pop.best_result())
       
        pop_next = pop.selection(pop.population)
        
        pop_next = list(map(pop.toolbox.clone, pop_next))
        # print(pop_next)

        #crossover
        pop_next = pop.crossover(pop_next, crossover)
        # print(pop_next)


        #mutating the population
        pop_next = pop.mutation(pop_next, mutation)
        # pop.printPop(pop_next)

        # evaluating new population
        # print(pop_next)
        pop.evaluation(pop_next, train_pos_target, 
                       train_neg_target, train_facts_target,
                       test_pos_target, test_neg_target, 
                       test_facts_target)

        # worst_individuals = pop.toolbox.selWorst(pop_next, 1)
        # for ind in worst_individuals:
        #     pop_next.remove(ind)

        # pop_next.extend(best_individuals)

        pop.population[:] = pop_next
    best_evaluates.append(pop.best_result())
    return pop, best_evaluates
