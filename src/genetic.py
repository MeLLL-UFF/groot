from deap import tools
from deap import base
from deap import creator

from src.individual import *
from src.population import *
from src.transfer import *

def genetic(src_struct, target, source, pred_inst, pos_target, 
            neg_target, facts_target, NUM_GEN=600, 
            pop_size=10, crossover=0.6, mutation=0.3):

    pop = Population(pop_size)
    best_evaluates = []

    
    pop.construct_population(src_struct, target, source, pred_inst)
    
    pop.evaluation(pop.population, pos_target, 
                   neg_target, facts_target)

    
    for generation in range(NUM_GEN):
        pred_inst.kb_source = pred_inst.kb_target
        pred_inst.generate_new_preds()

        for ind in pop.population:
            ind.source_tree = ind.individual_trees
            ind.predicate_inst.kb_source = ind.predicate_inst.kb_target
            ind.source = ind.target
            ind.predicate_inst.mapping_var = {}
      
        best_individuals = pop.toolbox.selBest(pop.population, 1)

        best_evaluates.append(pop.best_result())
       
        pop_next = pop.selection(pop.population)
        
        pop_next = list(map(pop.toolbox.clone, pop_next))
       

        #crossover
        pop_next = pop.crossover(pop_next, crossover)

        #mutating the population
        pop_next = pop.mutation(pop_next, mutation)

        # evaluating new population
        pop.evaluation(pop_next, pos_target, 
                       neg_target, facts_target)

        worst_individuals = pop.toolbox.selWorst(pop_next, 1)
        for ind in worst_individuals:
            pop_next.remove(ind)

        pop_next.extend(best_individuals)

        pop.population[:] = pop_next
    best_evaluates.append(pop.best_result())

    return pop, best_evaluates
