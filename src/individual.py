from boostsrl import boostsrl
import copy
from random import randint
import re
import string


class Individual:

	def __init__(self, tree_source, bk_source, bk_target, pred_target, target):
		#pred_target é : pred_target = [('movie', '+,-'), ('director', '+'),...]
		self.tree_source = tree_source
		self.modified_tree_source = tree_source
		self.bk_source = bk_source
		self.bk_target = bk_target
		self.pred_target = pred_target
		self.target = target
		self.variables = []
		self.new_bk_source = []
		self.new_bk_target = []
		self.transfer = []

	def compare_predicates(self, bk, pred_source, pred_target):
		for pred in bk:
			if pred_target in pred:
				if len(pred.split('(')[1].split(',')) == len(pred_source.split('(')[1].split(',')):
					return '({}'.format(pred_source.split('(')[1])
				else:
					#COLOCAR MENSAGEM DE ERRO
					return False
		#COLOCAR MENSAGEM DE ERRO
		return False

	def change_predicate(self, pred, new_info):
		#new_info é uma lista com o que se quer colocar entre pred e as variáveis
		#pred é do tipo: predicado(A)
		new_info = "".join(str(x) for x in new_info)
		return '{}{}({}'.format(pred.split('(')[0], new_info, pred.split('(')[1])

	def mapping_transfer(self, source_pred, target_pred, tree_number, index_node):
		#source é do tipo: pred(A) e target é pred
		new_source_pred = source_pred.split('(')[0]
		no_var = len(source_pred.split('(')[1].split(','))
		for pred in self.new_bk_source:
			if new_source_pred in pred and 'source: {}'.format(pred) not in self.transfer: 
				self.transfer.append('source: {}'.format(pred))
				break
		for pred in self.new_bk_target:
			if target_pred in pred and 'target: {}'.format(pred) not in self.transfer:
				self.transfer.append('target: {}'.format(pred))
				break
		var_pred = '(A'
		for i in range(1, no_var-1):
			var_pred = '{},{}'.format(var_pred, list(string.ascii_uppercase)[i])
		if no_var > 1:
			var_pred = '{},{})'.format(var_pred, list(string.ascii_uppercase)[no_var-1])
		else: 
			var_pred = '{})'.format(var_pred)
		if 'setMap: {}={}{}.'.format(self.change_predicate(source_pred, [tree_number, index_node]),
											target_pred, var_pred) not in self.transfer:
			self.transfer.append('setMap: {}={}{}.'.format(self.change_predicate(source_pred, 
														    [tree_number, index_node]),
															target_pred, var_pred))

	def generate_new_preds(self):
		self.new_bk_source = copy.deepcopy(self.bk_source)
		self.new_bk_target =copy.deepcopy(self.bk_target)

		for index in range(0, len(self.new_bk_source)):
		    self.new_bk_source[index] = self.new_bk_source[index].replace('+', '')
		    self.new_bk_source[index] = self.new_bk_source[index].replace('-', '')

		for index in range(0, len(self.new_bk_target)):
		    self.new_bk_target[index] = self.new_bk_target[index].replace('+', '')
		    self.new_bk_target[index] = self.new_bk_target[index].replace('-', '')

	def get_branch(self, curr_value, next_value):
		if curr_value == '': 
			return next_value
		return '{},{}'.format(curr_value, next_value)

	def get_modes(self, structured_tree, pred_source):
		#LEVA EM CONSIDERAÇAO QUE SÓ HÁ UM PREDICADO EM CADA NÓ
		new_pred_source = '{}('.format(pred_source.split('(')[0])
		if not self.variables:
			self.variables.extend(("".join(re.findall('\((.*?)\)', structured_tree[0]))).split(', '))
		tmp_var = ("".join(re.findall('\((.*?)\)',pred_source))).split(',')
		for var in tmp_var:
			if var in self.variables:
				new_pred_source += '+,'
			else: 
				new_pred_source += '-,'
				self.variables.append(var)
		return '{})'.format(new_pred_source) 

	def get_valid_predicates(self, pred_source):
		#list_pred é : list_pred = [('movie', '+,-'), ('director', '+'),...]
		#pred_source é: 'movie(+person,-person).' --> NAO VEM ASSIM: verificar quais variáveis já estão na árvore, se já tiver a variável, é +, cc -
		#VERIFICAR CASO ONDE HÁ DOIS PREDICADOS NO PRED_SOURCE
		occur_modes = ','.join([pred_source[occur.start()] 
			               for occur in re.finditer('[+\-]', pred_source)])
		valid_pred = []
		for pred, mode in self.pred_target:
			if mode == occur_modes: valid_pred.append(pred)
		return valid_pred

	def change_pred(self, source_pred, target_pred):
		#pegar os parâmetros do source_pred e passar para o target_pred
		return '{}({}'.format(target_pred.split('(')[0], source_pred.split('(')[1])

	def define_individual(self, structured_tree, tree_number):
		individual_tree = []
		target = structured_tree[0]
		nodes = structured_tree[1]
		for values, node in nodes.items():
			if values == '': 
				branch = '{} :- {}.'.format(target, node)
			else: branch = '{}.'.format(node)
			left_branch = 'true' if self.get_branch(values, 'true') in nodes  else 'false'
			right_branch = 'true' if self.get_branch(values, 'false') in nodes else 'false'
			individual_tree.append('{};{};{};{};{}'.format(tree_number, values, 
														branch, left_branch, right_branch))
		return individual_tree

	def define_individuals(self, structured_trees):
		individual_trees = []
		for index in range(0, len(structured_trees)):
			individual_trees.extend(self.define_individual(structured_trees[index], index))
		return individual_trees

	def generate_random_individual(self, structured_tree, tree_number):
		new_tree = []
		# new_tree.append(structured_tree[0])
		index_node = 1
		modes = self.compare_predicates(self.bk_target, structured_tree[0], self.target)
		if modes:
			new_tree.append('{}{}'.format(self.target, modes))
		self.mapping_transfer(structured_tree[0], self.target, tree_number, index_node)
		for values, node in structured_tree[1].items():
			#mode: pegar do bk
			#num_variaveis: pegar do bk
			# guardar do anterior para pegar no proximo e nao jogar fora a list_pred
			#list_pred = [movie(+movie, -person), director(+person)] e, ao pegar mode e num_variaveis
			# pegar apenas os predicados que casam fazendo uma variavel allowed_pred
			#choice = random em allowed_pred
			#trocar allowed_pred[choice] e node
			pred_source = self.get_modes(structured_tree, node)
			valid_preds = self.get_valid_predicates(pred_source)
			index_choice = randint(0, len(valid_preds)-1)
			new_node = self.change_pred(node, valid_preds[index_choice])
			self.mapping_transfer(node, valid_preds[index_choice], tree_number, index_node)
			structured_tree[1][values] = node.replace(node, new_node)
			self.modified_tree_source[tree_number][1][values] = node.replace(node, 
																			self.change_predicate(node, 
																								 [tree_number, 
																								 index_node]))
			index_node += 1
		return structured_tree

	def generate_random_individuals(self):
		individual_structured_trees = []
		for index in range(0, len(self.tree_source)):
			structured_tree = copy.deepcopy(self.tree_source[index])
			individual_structured_trees.append(self.generate_random_individual(structured_tree, index))
		return individual_structured_trees

	def evaluate(self, individual_tree, target):
		pass
		# mapping_preds = self.generate_map(individual_tree)
		# background_tr = boostsrl.modes(self.bk_target, [target], useStdLogicVariables=False)
		# model_tr = boostsrl.train(background_tr, train_pos_target, train_neg_target, train_facts_target, refine=individual_tree, transfer=mapping_preds, trees=trees_number)
		#fazer teste e retornar

	def construct_individual(self):
		self.generate_new_preds()
		structured_trees = self.generate_random_individuals()
		individual_trees = self.define_individuals(structured_trees)
		return individual_trees

