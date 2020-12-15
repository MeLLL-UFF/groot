from src.individual import *
from deap import tools
from deap import base
from deap import creator
from random import random


class Population:

    def __init__(self, pop_size=20):
        self.population = []
        self.pop_size = pop_size
        self.toolbox = base.Toolbox()
        self.toolbox.register("mate", tools.cxOnePoint) 
        # self.hof = tools.HallOfFame(2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("selBest", tools.selBest)
        self.toolbox.register("selWorst", tools.selWorst)

    def construct_population(self, tree_source, target, source, predicate_inst):
        for index in range(self.pop_size):

            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", Individual, fitness=creator.FitnessMin)
            tmp = creator.Individual(tree_source, target, source, predicate_inst)

            # self.toolbox.register("function", tmp.constructIndividual, list_flags)
            self.toolbox.register("function", tmp.generate_individuals)
            self.toolbox.function()
            self.population.append(tmp)

    def print_pop(self):
        print ('['),
        for ind in self.population:
            print (ind.individual_trees)
        print (']')

    #selecting population
    def selection(self, population):
        '''
            Select the population for the next generation according to definitions on "select"
        '''
        return self.toolbox.select(population, self.pop_size)

    #mutating population
    def mutation(self, new_pop, mutation_rate):
        '''
            Making mutation in a population
        '''
        pop = []
        for individual in new_pop:
            individual_aux = individual.mutation(individual, mutation_rate)
            pop.append(individual_aux)
        return pop

    def crossover(self, new_pop, cross_rate):
        '''
            Making crossover between individuals from the same population
        '''
        for part1, part2 in zip(new_pop[::2], new_pop[1::2]):
            if random() < cross_rate:     
                part1_range = list(range(0, 10))
                part2_range = list(map(lambda x:x+10, part1_range))
                div_part1, div_part2 = self.toolbox.mate(part1_range, part2_range)
                part1, part2 = part1.crossover(part1, part2, div_part1, div_part2)
        return new_pop

    #evaluating population
    def evaluation(self, new_pop, pos_target, 
                   neg_target, facts_target):
        '''
            Evaluating all population
        '''
        self.toolbox.register("evaluate", new_pop[0].evaluate, pos_target=pos_target, 
                        neg_target=neg_target, facts_target=facts_target)
        fitnesses = map(self.toolbox.evaluate, new_pop) #<--- problema
        
        for ind, fit in zip(new_pop, fitnesses):
            ind.fitness.values = fit

    #find best evaluate for the population
    def best_result(self):
        '''
            Find the best result in population
            The best result depends on the goal 
            Here, the goal is maximize the evaluate of the individual
        '''
        result = self.population[0].fitness.values[0]
        for indice in range(self.pop_size):
            fit = self.population[indice].fitness.values
            print(fit, fit[0] < result)
            if fit[0] < result:
                result = fit[0]
        print(f"bestResult: {result}")
        return result

