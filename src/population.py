from src.individual import *
from deap import tools
from deap import base
from deap import creator


class Population:

	def __init__(self, pop_size=5):
		self.population = []
		self.pop_size = pop_size
		self.toolbox = base.Toolbox()
		# self.toolbox.register("mate", tools.cxTwoPoint) 

	def construct_population(self, tree_source, bk_source, bk_target, pred_target,
		                     target):
		for index in range(self.pop_size):

			creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
			creator.create("Individual", Individual, fitness=creator.FitnessMax)
			tmp = creator.Individual(tree_source, bk_source, bk_target, 
				                     pred_target, target)

			# self.toolbox.register("function", tmp.constructIndividual, list_flags)
			self.toolbox.register("function", tmp.construct_individual)
			self.toolbox.function()
			self.population.append(tmp)

	def print_pop(self):
		print ('['),
		for ind in self.population:
			print (ind)
		print (']')






