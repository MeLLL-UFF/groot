from src.individual import *
from deap import tools
from deap import base
from deap import creator
from random import random


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

    def construct_population(self, source_tree, target, source, predicate_inst):
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
            tmp = creator.Individual(source_tree, target, source, predicate_inst)

            # self.toolbox.register("function", tmp.constructIndividual, list_flags)
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
            print (ind.individual_trees)
        print (']')

    def selection(self, population):
        """
            Select the population for the next generation according to definitions on "select"

            Parameters
            ----------
            population: list with Individual instances

            Returns
            ----------
            population: list, of pop_size size, with selected Individual instances
        """
        return self.toolbox.select(population, self.pop_size)

    def mutation(self, population, mutation_rate):
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
            individual_aux = individual.mutation(individual, mutation_rate)
            pop.append(individual_aux)
        return pop

    def crossover(self, population, cross_rate):
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
        for part1, part2 in zip(population[::2], population[1::2]):
            if random() < cross_rate:     
                part1_range = list(range(0, 10))
                part2_range = list(map(lambda x:x+10, part1_range))
                div_part1, div_part2 = self.toolbox.mate(part1_range, part2_range)
                part1, part2 = part1.crossover(part1, part2, div_part1, div_part2)
        return population

    def evaluation(self, population, pos_target, 
                   neg_target, facts_target):
        """
            Evaluating all population

            Parameters
            ----------
            population: list with Individual instances
            pos_target: list of lists
            neg_target: list of lists
            facts_target: list of lists

        """
        self.toolbox.register("evaluate", population[0].evaluate, pos_target=pos_target, 
                        neg_target=neg_target, facts_target=facts_target)
        fitnesses = map(self.toolbox.evaluate, population) #<--- problema
        
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

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
            print(fit, fit[0] < result)
            if fit[0] < result:
                result = fit[0]
        print(f"bestResult: {result}")
        return result

