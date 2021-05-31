import copy
from deap import tools, base, creator
from random import random, randint, uniform


from src.individual import *
from src.predicate import *


class Population:

    def __init__(self, pop_size=20):
        """
            Constructor

            Parameters
            ----------
            pop_size: int
        """
        self.population = []
        self.pop_size = pop_size
        self.toolbox = base.Toolbox()
        self.toolbox.register("mate", tools.cxOnePoint) 
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("selBest", tools.selBest)
        self.toolbox.register("selWorst", tools.selWorst)

    def construct_population(self, source_tree, target, source, kb_source,
                            kb_target, target_pred):
        """
            Construct the first population

            Parameters
            ----------
            source_tree: list of lists, containing the structure of the source tree,
                         serving as a base to the transfer
                         example:
                            [['0;;workedunder(A,B):-actor(A);false;true', '0;'...],
                             ['1;;workedunder(A,B):-director(A);false;true', '1;'...']]
            target: string
            source: string
            predicate_inst: instance of the class Predicate
        """
        for index in range(self.pop_size):

            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", Individual, fitness=creator.FitnessMin)
            predicate_inst = Predicate(kb_source, kb_target, target_pred)
            tmp = creator.Individual(source_tree, target, source, predicate_inst)

            self.toolbox.register("function", tmp.generate_individuals)
            self.toolbox.function()
            self.population.append(tmp)

    def print_pop(self):
        """
            Printing the entire population

            This method needs improvements
        """
        print ('['),
        for ind in self.population:
            print(ind.individual_trees)
            print(ind.fitness)
        print (']')

    def mutation(self, population, mutation_rate, revision):
        """
            Making mutation in the population

            Parameters
            ----------
            population: list with Individual instances
            mutation_rate: float

            Returns
            ----------
            pop: list with Individual instances
        """
        pop = []
        for individual in population:
            if random() <= mutation_rate:
                if revision:
                    individual_aux = individual.mutation_revision(individual, mutation_rate, revision)
                else:
                    individual_aux = individual.mutation(individual, mutation_rate)
                individual_aux.need_evaluation = True
                pop.append(individual_aux)
            else:
                pop.append(individual)
        return pop

    # def _get_genes(self, individual_pop, individual_elite, crossover_rate):
    #     positions = []
    #     new_ind = copy.deepcopy(individual_elite)
    #     for i in range(0, len(individual_pop.individual_trees)):
    #         positions.append(uniform(0.0, 1.0))

    #     new_ind.individual_trees = []
    #     new_ind.source_tree = []
    #     for idx in range(0, len(positions)):
    #         if positions[idx] <= crossover_rate:
    #             new_ind.individual_trees.append(individual_elite.individual_trees[idx])
    #             new_ind.source_tree.append(individual_elite.source_tree[idx])
    #         else:
    #             new_ind.individual_trees.append(individual_pop.individual_trees[idx])
    #             new_ind.source_tree.append(individual_pop.source_tree[idx])
    #     new_ind.need_evaluation = True
    #     return new_ind

    def _get_genes(self, individual_pop, individual_elite, crossover_rate):
        new_ind = copy.deepcopy(individual_elite)

        new_ind.predicate_inst.kb_source.extend(individual_pop.predicate_inst.kb_source)
        new_ind.predicate_inst.new_kb_source.extend(individual_pop.predicate_inst.new_kb_source)
        new_ind.predicate_inst.new_first_kb_source.extend(individual_pop.predicate_inst.new_first_kb_source)
   

        new_ind.individual_trees = []
        new_ind.first_source_tree = []
        
        for tree in range(0, len(individual_elite.individual_trees)):
            base_individual_trees = len(individual_pop.individual_trees[tree])
            if len(individual_pop.individual_trees[tree]) < len(individual_elite.individual_trees[tree]):
                base_individual_trees = len(individual_elite.individual_trees[tree])
            positions = []
            for i in range(0, base_individual_trees):
                positions.append(uniform(0.0, 1.0))

            res = new_ind.crossover_genes(individual_pop.individual_trees[tree], individual_elite.individual_trees[tree], 
                                    individual_pop.first_source_tree[tree], individual_elite.first_source_tree[tree], 
                                    positions, crossover_rate)
            new_ind.individual_trees.append(res[0])   
            new_ind.first_source_tree.append(res[1])  
                            
        new_ind.need_evaluation = True
        return new_ind


    def crossover(self, population, elite, len_pop, cross_rate):
        """
            Making crossover between individuals from the same population

            Parameters
            ----------
            population: list with Individual instances
            cross_rate: float

            Returns
            ----------
            population: list with Individual instances
        """
        new_population = copy.deepcopy(population)
        new_elite = copy.deepcopy(elite)
        new_pop = []
        while len(new_pop) < len_pop:
            # if random() < cross_rate: 
            pos_population = randint(0, len(new_population)-1)
            pos_elite = randint(0, len(new_elite)-1)  
            new_pop.append(self._get_genes(new_population[pos_population], 
                                           new_elite[pos_elite], 
                                           cross_rate))
            # part1_range = list(range(0, 10))
            # part2_range = list(map(lambda x:x+10, part1_range))
            # div_part1, div_part2 = self.toolbox.mate(part1_range, part2_range)
            # new_ind1, new_ind2 = new_population[pos_population].crossover(new_population[pos_population], new_elite[pos_elite], div_part1, div_part2)
            # new_ind1.need_evaluation = True
            # new_ind2.need_evaluation = True
            # new_pop.append(new_ind1)
            # new_pop.append(new_ind2)
        return new_pop

    def evaluation(self, population, trees, pos_target, 
                   neg_target, facts_target):
        """
            Evaluating all population

            Parameters
            ----------
            population: list with Individual instances
            trees: int
            pos_target: list of lists
            neg_target: list of lists
            facts_target: list of lists

        """
        evaluate_pop = [] 
        i = 0
        for individual in population:
            if individual.need_evaluation:
                evaluate_pop.append(individual)

        for individual in population:
            individual.need_evaluation = False

        if len(evaluate_pop) > 0:
            self.toolbox.register("before_evaluate", evaluate_pop[0].before_evaluate)

            trees = map(self.toolbox.before_evaluate, evaluate_pop)

            for ind, tree in zip(evaluate_pop, trees):
                ind.individual_trees = tree[0]
                ind.modified_src_tree = tree[1]
                ind.transfer = tree[2]

            results = evaluate_pop[0].run_evaluate(evaluate_pop, pos_target, neg_target, facts_target)
            for ind, result in zip(evaluate_pop, results):
                ind.fitness.values = result[3],
                ind.results.append(result[1])    
                ind.variances = result[2] 

    def best_result(self):
        """
            Find the best result in population
            The best result depends on the goal 
            Here, the goal is minimize the evaluate of the individual

            Returns
            ----------
            result: float
        """
        result = self.population[0].fitness.values[0]
        for indice in range(self.pop_size):
            fit = self.population[indice].fitness.values
            if fit[0] < result:
                result = fit[0]
        # print(f"bestResult: {result}")
        return result

    def sel_best_cll(self, best_ind_auc_pr): 
        best_auc_pr = best_ind_auc_pr.results[-1]['m_auc_pr']
        best_inds = []
        for i in self.population:
            if i.results[-1]['m_auc_pr'] == best_auc_pr:
                best_inds.append(i)
        best_cll = best_inds[0].results[-1]['m_cll']
        best_ind = best_inds[0]
        for i in best_inds:
            if i.results[-1]['m_cll'] < best_cll:
                best_ind = i
                best_cll = i.results[-1]['m_cll']
        return [best_ind]

    def get_all_best_results(self):
        """
            Find the best result in population, with all components (cll, precision, auc roc etc)
            The best result depends on the goal 
            Here, the goal is minimize the evaluate of the individual

            Returns
            ----------
            result: float
        """
        result_auc_pr = self.population[0].fitness.values[0]
        result = self.population[0].results[-1]
        for indice in range(self.pop_size):
            fit = self.population[indice].fitness.values
            # print(fit, fit[0] < result)
            if fit[0] < result_auc_pr:
                # result = self.population[indice].results[-1]
                result_auc_pr = fit[0]

        all_best = []
        for indice in range(self.pop_size):
            fit = self.population[indice].fitness.values
            if fit[0] == result_auc_pr:
                print(fit[0], result_auc_pr)
                all_best.append(self.population[indice].results[-1])
        # print(all_best)
        best_cll = all_best[0]['m_cll']
        result = all_best[0]
        for best in all_best:
            if best['m_cll'] < best_cll:
                result = best
                best_cll= best['m_cll']
        # print(f"RESULT: {result}")
        return result

