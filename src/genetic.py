from deap import tools, base, creator


from src.individual import *
from src.population import *
from src.transfer import *


def genetic(src_struct, target, source, pos_target, 
            neg_target, facts_target, kb_source, kb_target,
            target_pred, NUM_GEN=600, pop_size=10, 
            crossover=0.6, mutation=0.3):

    pop = Population(pop_size)
    best_evaluates = []
    all_best_results = []

    has_same_best_value = 0
    pop.construct_population(src_struct, target, source, kb_source,
                            kb_target, target_pred)
    
    pop.evaluation(pop.population, pos_target, 
                   neg_target, facts_target)


    for generation in range(NUM_GEN):
        print("GENERATION: ", generation)

        for ind in pop.population:
            ind.source_tree = ind.individual_trees
            ind.predicate_inst.kb_source = ind.predicate_inst.kb_target
            ind.source = ind.target
            ind.predicate_inst.generate_new_preds()
            if len(ind.results) < NUM_GEN:
                ind.results.append(ind.results[-1])
            ind.predicate_inst.mapping_var = {}
      
        best_individuals = pop.toolbox.selBest(pop.population, 1)

        # if len(best_evaluates) > 0 and pop.best_result() == best_evaluates[-1]:
        #     has_same_best_value += 1
        # else:
        #     has_same_best_value = 0
        best_evaluates.append(pop.best_result())
        print('MELHOR RESULTADO: ', pop.best_result())

        # if has_same_best_value == ((NUM_GEN)/2)+1:
        #     return pop, best_evaluates, all_best_results
       
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

        all_best_results.append(pop.get_all_best_results())
        pop_next.extend(best_individuals)

        pop.population[:] = pop_next
    all_best_results.append(pop.get_all_best_results())
    best_evaluates.append(pop.best_result())

    return pop, best_evaluates, all_best_results
