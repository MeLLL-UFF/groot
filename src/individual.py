from boostsrl import boostsrl
import copy
from random import randint
from random import random
import re
import string
import shutil
import os

from src.transfer import Transfer


class Individual:

    def __init__(self, source_tree, target, predicate_instance):
        #pred_target é : pred_target = [('movie', '+,-'), ('director', '+'),...]
        #source_tree é [['0;;workedunder(A,B):-actor(A);false;true', '0;'...]]
        self.source_tree = source_tree
        self.modified_src_tree = []
        self.target = target
        self.predicate_inst = predicate_instance #<--- Predicate(...)
        self.transfer = Transfer(predicate_instance, target)
        self.individual_trees = []

    def generate_random_individual(self, src_tree, tree_number):
        new_individual_tree = []
        for line_index in range(0, len(src_tree)):
            if line_index == 0:
                new_src_pred = src_tree[line_index].split(";")
                src_pred = src_tree[line_index].split(':-')[0].split(";")[2]
                res = self.predicate_inst.new_pred(src_pred, self.target)
                new_src_pred[2] = f"{res}:-{new_src_pred[2].split(':-')[1]}"
                new_src_pred = ";".join(new_src_pred)
                pred_source = self.predicate_inst.get_modes(src_tree, src_tree[line_index].split(":-")[1])
            else: 
                pred_source = self.predicate_inst.get_modes(src_tree, src_tree[line_index])
                new_src_pred = src_tree[line_index]
            # print(pred_source)
            new_individual_tree.append(self.predicate_inst.change_pred(new_src_pred, pred_source))
        return new_individual_tree

    def generate_individuals(self):
        # print("ESTOU AQUI generate_individuals")
        self.predicate_inst.generate_new_preds()
        for tree_index in range(0, len(self.source_tree)):
            self.individual_trees.append(self.generate_random_individual(self.source_tree[tree_index], tree_index))

    # def construct_individual(self):
    #     self.generate_individuals()

    def evaluate(self, ind, train_pos_target, train_neg_target, train_facts_target,
                 test_pos_target, test_neg_target, test_facts_target):
        self.transfer = Transfer(ind.predicate_inst, ind.target)
        # print(self.transfer.predicate_inst.bk_source)
        # print("ESTOU EM evaluate")
        # shutil.rmtree('boostsrl/train')
        # os.remove('boostsrl/train_output.txt')
        # shutil.rmtree('boostsrl/test')
        # os.remove('boostsrl/test_output.txt')
        # os.remove('boostsrl/refine.txt')
        # os.remove('boostsrl/transfer.txt')
        # print(ind.source_tree)
        ind.modified_src_tree = ind.transfer.mapping_all_trees(ind.individual_trees, ind.source_tree)
        refine = []
        for tree in ind.modified_src_tree:
            refine.extend(tree)
        # [refine.extend(tree) for tree in self.modified_src_tree] #<--- VERIFICAR
        # print("TAMANHO TRANSFER: ", len(ind.transfer.transfer))
        background_train = boostsrl.modes(ind.predicate_inst.bk_target, [ind.target])
        model_tr = boostsrl.train(background_train, train_pos_target, train_neg_target, 
                                  train_facts_target, refine=refine, transfer=ind.transfer.transfer, 
                                  trees=20)
        test_model = boostsrl.test(model_tr, test_pos_target, test_neg_target, 
                                   test_facts_target, trees=20)
        return -test_model.summarize_results()['CLL'],

    def mutate_pred(self, individual_tree, mut_rate):
        new_individual_tree = []
        for pred in individual_tree:
            if random() < mut_rate:
                if len(pred.split(":-")) > 1:
                    pred_source = self.predicate_inst.get_modes(individual_tree, 
                                                                pred.split(":- ")[1])
                else:
                    pred_source = self.predicate_inst.get_modes(individual_tree, pred)

                new_individual_tree.append(self.predicate_inst.change_pred(pred, pred_source))
            else:
                new_individual_tree.append(pred)
        return new_individual_tree

    def mutation(self, ind, mut_rate):
        new_individual_trees = []
        for individual_tree in ind.individual_trees:
            new_individual_trees.append(ind.mutate_pred(individual_tree, mut_rate))
        ind.individual_trees = new_individual_trees
        return ind

    def form_individual(self, tree_one, tree_two, div, threshold):
        new_tree = []
        for index in div:
            if index-1 < threshold:
                new_tree.append(tree_one[(index-1)%10])
            else:
                new_tree.append(tree_two[(index-1)%10])
        return new_tree

    def crossover(self, tree_one, tree_two, div_one, div_two):
        tree_one.individual_trees = self.form_individual(tree_one.individual_trees, 
                                                         tree_two.individual_trees, 
                                                         div_one, 
                                                         len(tree_one.individual_trees))
        tree_two.individual_trees = self.form_individual(tree_one.individual_trees, 
                                                         tree_two.individual_trees, 
                                                         div_two, 
                                                         len(tree_one.individual_trees))

        tree_one.source_tree = self.form_individual(tree_one.source_tree, 
                                                    tree_two.source_tree, 
                                                    div_one, 
                                                    len(tree_one.source_tree))
        tree_two.source_tree = self.form_individual(tree_one.source_tree, 
                                                    tree_two.source_tree, 
                                                    div_two, 
                                                    len(tree_one.source_tree))
        return tree_one, tree_two
       


