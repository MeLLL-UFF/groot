from boostsrl import boostsrl
import copy
import multiprocessing
import numpy as np
import os
from random import choice, randint, random
import re
import shutil
import string
import sys


from src.transfer import Transfer
from src.revision import Revision


class Individual:

    def __init__(self, source_tree, target, source, predicate_instance):
        #pred_target é : pred_target = [('movie', '+,-'), ('director', '+'),...]
        #source_tree é [['0;;workedunder(A,B):-actor(A);false;true', '0;'...]]
        """
            Constructor

            Parameters
            ----------
            source_tree: list of lists, containing the structure of the source tree,
                         serving as a base to the transfer
                         example:
                            [['0;;workedunder(A,B):-actor(A);false;true', '0;'...],
                             ['1;;workedunder(A,B):-director(A);false;true', '1;'...']]
            target: string
            source: string
            predicate_instance: instance of the class Predicate
        """
        self.first_source_tree = source_tree
        self.source_tree = source_tree
        self.modified_src_tree = []
        self.target = target
        self.source = source
        self.predicate_inst = predicate_instance #<--- Predicate(...)
        self.transfer = Transfer(predicate_instance, target)
        self.individual_trees = []
        self.need_evaluation = True
        self.results = []
        self.revision = Revision()
        self.variances = None

    def generate_random_individual(self, src_tree, tree_number):
        """
            Generate one individual randomly
            Changes the predicate of the source tree to predicates from the
            target knowledge base
    
            Parameters
            ----------
            src_tree: source tree where to change the predicates
            tree_number: number of the tree       

            Returns
            ----------
            new_individual_tree: tree with the same structure of src_tree but with predicates
                                from the target knowledge base
        """
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
            new_individual_tree.append(self.predicate_inst.change_pred(self.source, self.target, new_src_pred, pred_source))
        return new_individual_tree

    def generate_individuals(self):
        """
            Generate all individuals randomly
        """
        self.predicate_inst.generate_new_preds()
        for tree_index in range(0, len(self.source_tree)):
            self.individual_trees.append(self.generate_random_individual(self.source_tree[tree_index], tree_index))

    @staticmethod
    def define_splits(pos_target, neg_target, facts_target, fold_test_number):
        """
            Defining folds to the train and test, according to the fold test number

            Parameters
            ----------
            pos_target: list of lists
            neg_target: list of lists
            facts_target: list of lists
            fold_test_number: int

            Returns
            ----------
            train_pos_target: list
            train_neg_target: list
            train_facts_target: list
            test_pos_target: list
            test_neg_target: list
            test_facts_target: list
        """
        train_pos_target = []; test_pos_target = []
        train_neg_target = []; test_facts_target = []
        train_facts_target = []; test_neg_target = []
        for index in range(0, len(pos_target)):
            if index == fold_test_number:
                train_pos_target = pos_target[index]
                train_neg_target = neg_target[index]
                train_facts_target = facts_target[index]
            else:
                test_pos_target.extend(pos_target[index])
                test_neg_target.extend(neg_target[index])
                test_facts_target.extend(facts_target[index])
        return train_pos_target, train_neg_target, train_facts_target, \
               test_pos_target, test_neg_target, test_facts_target

    @staticmethod
    def get_results(results):
        """
            Get results from dictionary

            Parameters
            ----------
            results: dictionary

            Returns
            ----------
            mean_auc_pr: float
            mean_auc_roc: float
            mean_cll: float
            mean_precision: float
            mean_recall: float
            mean_f1: float
            std_auc_pr: float
            std_auc_roc: float
            std_cll: float
            std_precision: float
            std_recall: float
            std_f1: float
        """
        auc_pr = [result['AUC PR'] for result in results]
        auc_roc = [result['AUC ROC'] for result in results]
        cll = [result['CLL'] for result in results]
        prec = [result['Precision'] for result in results]
        rec = [result['Recall'] for result in results]
        f1 = [result['F1'] for result in results]
        return np.mean(auc_pr), np.mean(auc_roc), np.mean(cll), np.mean(prec), \
               np.mean(rec), np.mean(f1), np.std(auc_pr), np.std(auc_roc), np.std(cll), \
               np.std(prec), np.std(rec), np.std(f1)

    def print_results(self, auc_pr, auc_roc, cll, prec, rec, f1, type='MEDIA'):
        """
            Printing the results
            
            Parameters
            ----------
            auc_pr: float
            auc_roc: float
            cll: float
            prec: float
            rec: float
            f1: float
            type: string (can be MEDIA or STD)

        """
        print(type)
        print("AUC PR: ", auc_pr)
        print("AUC ROC: ", auc_roc)
        print("CLL: ", cll)
        print("PREC: ", prec)
        print("RECALL: ", rec)
        print("F1: ", f1)
        print("-------------------")

    @staticmethod
    def _create_new_folder(idx):
        """
            Creating folders to run the individual of index idx

            It's necessary when doing training and testing in parallel

            Parameters
            ----------
            idx: int

        """
        if not os.path.exists(f'individual_{idx}/boostsrl'):
            os.makedirs(f'individual_{idx}/boostsrl')
        shutil.copy('boostsrl/auc.jar', f'individual_{idx}/boostsrl/')
        shutil.copy('boostsrl/v1-0.jar', f'individual_{idx}/boostsrl/')

    @staticmethod
    def _delete_folder(idx):
        """
            Deleting folder created in create_new_folder method

            It's necessary when doing training and testing in parallel

            Parameters
            ----------
            idx: int
        """
        shutil.rmtree(f'individual_{idx}', ignore_errors=True)

    def before_evaluate(self, ind):
        """
            Checking if the transfer is reasonable

            Parameters
            ----------
            ind: Individual instance

            Returns
            ----------
            individual_trees: list of lists
            modified_src_tree: list of lists
            transfer: list
        """
        ind.transfer = Transfer(ind.predicate_inst, ind.target)
        ind.individual_trees = ind.predicate_inst.check_trees(ind)
        ind.modified_src_tree, ind.transfer.transfer = ind.transfer.mapping_all_trees(ind.individual_trees, ind.first_source_tree, ind)
        return ind.individual_trees, ind.modified_src_tree, ind.transfer

    @staticmethod
    def get_cll():
        f = open(f'boostsrl/test_output.txt').read()
        import re
        line = re.findall(f'%Pos.*|%Neg.*|%LL.*|%LL*', f)
        return [word.replace(' ','').replace('\t','').replace('%','') for word in line]
        
    @staticmethod
    def evaluate(args):
        """
            Evaluating the individual 
            This method will run in parallel
            The method also receives the data to train and test the transfer

            Parameters
            ----------
            args: dictionary
                  The keys need to be:
                  idx: Individual index in the population (int)
                  transfer: the mapping between source and target (list)
                  modified_src_tree: list of lists
                  pos_target: list of lists
                  neg_target: list of lists
                  facts_target: list of lists
                  target: target predicate to test (string)
                  kb_target: list

            Returns
            ----------
            -m_cll: tuple (the cll is a negative value; the objetive is to minimize the -mean(cll))
            results: dictionary
        """        
        pos_target = args['pos_target']
        neg_target = args['neg_target']
        facts_target = args['facts_target']
        transfer = args['transfer']
        # trees = args['trees']

        os.chdir(f'individual_{args["idx"]}')
# 
        refine = []
        for tree in args['modified_src_tree']:
            refine.extend(tree)

        background_train = boostsrl.modes(args['kb_target'], [args['target']], useStdLogicVariables=False, 
                                          maxTreeDepth=3, nodeSize=2, numOfClauses=8)
        results = []
        best_cll = 0.0
        for i in range(0, len(pos_target)):
            train_pos_target, train_neg_target, train_facts_target, \
            test_pos_target, test_neg_target, test_facts_target = Individual.define_splits(pos_target, neg_target, 
                                                                                     facts_target, i)
            train_neg_target = np.random.choice(train_neg_target, 2*len(train_pos_target))

            # print(f"Train facts: {len(train_facts_target)}")
            # print(f"Train neg: {len(train_neg_target)}")
            # print(f"Train pos: {len(train_pos_target)}")

            # print(f"Test facts: {len(test_facts_target)}")
            # print(f"Test neg: {len(test_neg_target)}")
            # print(f"Test pos: {len(test_pos_target)}")

            model_tr = boostsrl.train(background_train, train_pos_target, train_neg_target, 
                                      train_facts_target, refine=refine, transfer=transfer, 
                                      trees=10)

            # structured_src = []
            # for j in range(0, 10):
            #     try:
            #         structured_src.append(model_tr.get_structured_tree(treenumber=j+1).copy())
            #     except:
            #         structured_src.append(model_tr.get_structured_tree(treenumber='combine').copy())
            
            test_model = boostsrl.test(model_tr, test_pos_target, test_neg_target, 
                                       test_facts_target, trees=10)
            results_fold = test_model.summarize_results()
            if results_fold['CLL'] < best_cll:
                variances = [model_tr.get_variances(treenumber=i+1) for i in range(10)]
            results.append(results_fold)
        m_auc_pr, m_auc_roc, m_cll, m_prec, m_rec, \
        m_f1, s_auc_pr, s_auc_roc, s_cll, s_prec, s_rec, s_f1 = Individual.get_results(results)
        
       


        # print('MEDIA')
        # print("AUC PR: ", m_auc_pr)
        # print("AUC ROC: ", m_auc_roc)
        # print("CLL: ", m_cll)
        # print("PREC: ", m_prec)
        # print("RECALL: ", m_rec)
        # print("F1: ", m_f1)
        # print("-------------------")
        # print('STD')
        # print("AUC PR: ", s_auc_pr)
        # print("AUC ROC: ", s_auc_roc)
        # print("CLL: ", s_cll)
        # print("PREC: ", s_prec)
        # print("RECALL: ", s_rec)
        # print("F1: ", s_f1)
        # print("-------------------")


        results = {'m_auc_pr': m_auc_pr, 'm_auc_roc': m_auc_roc,
                   'm_cll': m_cll, 'm_rec': m_rec, 'm_pred': m_prec, 'm_f1': m_f1,
                   's_auc_pr': s_auc_pr, 's_auc_roc': s_auc_roc,
                   's_cll': s_cll, 's_rec': s_rec, 's_prec': s_prec, 's_f1': s_f1}
        os.chdir('..')
        return m_cll, results, variances, -m_auc_pr

    def _input_list(self, population, pos_target, neg_target, facts_target):
        """
            Input list to pass to evaluate
            Parameters
            ----------
            population: list
            pos_target: list
            neg_target: list
            facts_target: list

            Returns
            ----------
            input_list: list with dictionaries
        """   
        input_list = []
        for i in range(len(population)):
            input_list.append({'idx': i,
                               'transfer': population[i].transfer.transfer,
                               'kb_target': population[i].predicate_inst.kb_target,
                               'pos_target': pos_target,
                               'neg_target': neg_target,
                               'facts_target': facts_target,
                               'target': population[i].target,
                               'modified_src_tree': population[i].modified_src_tree})
        return input_list

    def run_evaluate(self, population, pos_target, neg_target, facts_target):
        """
            Run evaluation in parallel
            Before running the evaluation, create a folder for each individual
            After, delete all folders 

            Parameters
            ----------
            population: list
            trees: int
            pos_target: list
            neg_target: list
            facts_target: list

            Returns
            ----------
            results: list with tuples
        """  
        pool = multiprocessing.Pool()
        
        res = pool.map(self._create_new_folder, range(len(population)))
        pool.terminate()
        pool.join()

        input_list = self._input_list(population, pos_target, neg_target, facts_target)
        pool = multiprocessing.Pool()
        results = pool.map(self.evaluate, input_list)
        
        pool.terminate()
        pool.join()
        
        pool = multiprocessing.Pool()
        
        res = pool.map(self._delete_folder, range(len(population)))
        pool.terminate()
        pool.join()

        return results

    def mutate_pred(self, individual_tree, mut_rate):
        """
            Change each predicate from individual_tree according to the mutation rate
            The predicate will be changed to another predicate from the same knowledge base

            Parameters
            ----------
            individual_tree: list
            mut_rate: float

            Returns
            ----------
            new_individual_tree: list
        """
        new_individual_tree = []
        for pred in individual_tree:
            if random() <= mut_rate:
                if len(pred.split(":-")) > 1:
                    pred_source = self.predicate_inst.get_modes(individual_tree, 
                                                                pred.split(":- ")[1])
                else:
                    pred_source = self.predicate_inst.get_modes(individual_tree, pred)
                res = self.predicate_inst.change_pred(self.source, self.target, pred, pred_source)
                new_individual_tree.append(res)
            else:
                new_individual_tree.append(pred)
        return new_individual_tree

    def mutation(self, ind, mut_rate):
        """
            Making mutation in an individual according to the mutation rate

            Parameters
            ----------
            ind: Individual instance
            mut_rate: float

            Returns
            ----------
            ind: Individual instance
        """
        new_individual_trees = []
        for individual_tree in ind.individual_trees:
            new_individual_trees.append(ind.mutate_pred(individual_tree, mut_rate))
        ind.individual_trees = new_individual_trees
        return ind

    def mutation_revision(self, ind, mut_rate, revision):
        """
            Making mutation in an individual according to the mutation rate

            Parameters
            ----------
            ind: Individual instance
            mut_rate: float

            Returns
            ----------
            ind: Individual instance
        """
        if revision == 'random':
            random_line = True
        elif revision == 'guided':
            random_line = False
        else:
            possibles = [True, False]
            random_line = choice(possibles)
        operators = ['expansion', 'pruning']
        new_individual_trees = []
        new_source_tree = []
        for idx in range(0, len(ind.individual_trees)):
            operator = choice(operators)
            new_src, new_ind = ind.revision.modify_tree(ind,
                                                        ind.individual_trees[idx], 
                                                        ind.variances[idx],
                                                        ind.first_source_tree[idx], 
                                                        operator,
                                                        random_line)
            new_source_tree.append(new_src)
            new_individual_trees.append(new_ind)
        # print("KB: ", ind.predicate_inst.kb_source)
        ind.individual_trees = new_individual_trees
        ind.first_source_tree = new_source_tree
        return ind

    def form_individual(self, tree_one, tree_two, div, threshold):
        """
            Combining two individuals according to the threshold

            Parameters
            ----------
            tree_one: Individual instance
            tree_two: Individual instance
            div: list with the individuals division 
            threshold: threshold to the division

            Returns
            ----------
            new_tree: list
        """
        new_tree = []
        for index in div:
            if index < threshold:
                new_tree.append(tree_one[(index)%10])
            else:
                new_tree.append(tree_two[(index)%10])
        return new_tree

    def crossover(self, tree_one, tree_two, div_one, div_two):
        """
            Crossover between two individuals

            Parameters
            ----------
            tree_one: Individual instance
            tree_two: Individual instance
            div_one: list containing the parts of the tree_one
            div_two: list containing the parts of the tree_two

            Returns
            ----------
            tree_one: Individual instance
            tree_two: Individual instance
        """
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
        tree_one.first_source_tree = self.form_individual(tree_one.first_source_tree, 
                                                    tree_two.first_source_tree, 
                                                    div_one, 
                                                    len(tree_one.first_source_tree))
        tree_two.first_source_tree = self.form_individual(tree_one.first_source_tree, 
                                                    tree_two.first_source_tree, 
                                                    div_two, 
                                                    len(tree_one.first_source_tree))
        tree_one.predicate_inst.kb_source.extend(tree_two.predicate_inst.kb_source)
        tree_one.predicate_inst.new_kb_source.extend(tree_two.predicate_inst.new_kb_source)
        tree_one.predicate_inst.new_first_kb_source.extend(tree_two.predicate_inst.new_first_kb_source)
        tree_one.predicate_inst.kb_source = list(set(tree_one.predicate_inst.kb_source))
        tree_one.predicate_inst.new_kb_source = list(set(tree_one.predicate_inst.new_kb_source))
        tree_one.predicate_inst.new_first_kb_source = list(set(tree_one.predicate_inst.new_first_kb_source))
        

        tree_two.predicate_inst.kb_source.extend(tree_one.predicate_inst.kb_source)
        tree_two.predicate_inst.new_kb_source.extend(tree_one.predicate_inst.new_kb_source)
        tree_two.predicate_inst.new_first_kb_source.extend(tree_one.predicate_inst.new_first_kb_source)
        tree_two.predicate_inst.kb_source = list(set(tree_two.predicate_inst.kb_source))
        tree_two.predicate_inst.new_kb_source = list(set(tree_two.predicate_inst.new_kb_source))
        tree_two.predicate_inst.new_first_kb_source = list(set(tree_two.predicate_inst.new_first_kb_source))
        return tree_one, tree_two

    def crossover_tree(self, tree_one, tree_two, tree_one_src, tree_two_src):
        if len(tree_one) == 1 or len(tree_two) == 1:
            return tree_one, tree_two, tree_one_src, tree_two_src
        tree_one_choice = randint(1, len(tree_one)-1)
        tree_two_choice = randint(1, len(tree_two)-1)

        path_tree_one = tree_one[tree_one_choice].split(';')[1]
        path_tree_two = tree_two[tree_two_choice].split(';')[1]
        number_tree_one = tree_one[tree_one_choice].split(';')[0]
        number_tree_two = tree_two[tree_two_choice].split(';')[0]
        lines_tree_one = []
        lines_tree_two = []
        for i in range(0, len(tree_one)):
            if tree_one[i].split(';')[1].startswith(path_tree_one):
                lines_tree_one.append(i)
        for j in range(0, len(tree_two)):
            if tree_two[j].split(';')[1].startswith(path_tree_two):
                lines_tree_two.append(j)

        tree_one[tree_one_choice].split(';')[1].replace(path_tree_one, path_tree_two)
        tree_two[tree_two_choice].split(';')[1].replace(path_tree_two, path_tree_one)

        tree_one_src[tree_one_choice].split(';')[1].replace(path_tree_one, path_tree_two)
        tree_two_src[tree_two_choice].split(';')[1].replace(path_tree_two, path_tree_one)        

        new_tree_one = tree_one[:tree_one_choice]
        new_tree_two = tree_two[:tree_two_choice]

        new_tree_one_src = tree_one_src[:tree_one_choice]
        new_tree_two_src = tree_two_src[:tree_two_choice]

        for j in lines_tree_two:
            tmp_tree_two = tree_two[j].split(';')
            tmp_tree_two[0] = number_tree_one
            tmp_tree_two[1] = tmp_tree_two[1].replace(path_tree_two, path_tree_one, 1)
            new_tree_one.append(';'.join(tmp_tree_two))

            tmp_tree_two = tree_two_src[j].split(';')
            tmp_tree_two[0] = number_tree_one
            tmp_tree_two[1] = tmp_tree_two[1].replace(path_tree_two, path_tree_one, 1)
            new_tree_one_src.append(';'.join(tmp_tree_two))
        for k in lines_tree_one:
            tmp_tree_one = tree_one[k].split(';')
            tmp_tree_one[0] = number_tree_two
            tmp_tree_one[1] = tmp_tree_one[1].replace(path_tree_one, path_tree_two, 1)
            new_tree_two.append(';'.join(tmp_tree_one))

            tmp_tree_one = tree_one_src[k].split(';')
            tmp_tree_one[0] = number_tree_two
            tmp_tree_one[1] = tmp_tree_one[1].replace(path_tree_one, path_tree_two, 1)
            new_tree_two_src.append(';'.join(tmp_tree_one))

        new_tree_one.extend(tree_one[lines_tree_one[-1]+1:])
        new_tree_two.extend(tree_two[lines_tree_two[-1]+1:])

        new_tree_one_src.extend(tree_one_src[lines_tree_one[-1]+1:])
        new_tree_two_src.extend(tree_two_src[lines_tree_two[-1]+1:])

        return new_tree_one, new_tree_two, new_tree_one_src, new_tree_two_src

    def crossover_trees(self, ind1, ind2, part1, part2):
        tree_one = ind1.individual_trees[part1] 
        tree_two = ind2.individual_trees[part2]
        tree_one_src = ind1.first_source_tree[part1]
        tree_two_src = ind2.first_source_tree[part2]

        if len(tree_one) == 1 or len(tree_two) == 1:
            return ind1, ind2
        tree_one_choice = randint(1, len(tree_one)-1)
        tree_two_choice = randint(1, len(tree_two)-1)

        path_tree_one = tree_one[tree_one_choice].split(';')[1]
        path_tree_two = tree_two[tree_two_choice].split(';')[1]
        number_tree_one = tree_one[tree_one_choice].split(';')[0]
        number_tree_two = tree_two[tree_two_choice].split(';')[0]
        lines_tree_one = []
        lines_tree_two = []
        for i in range(0, len(tree_one)):
            if tree_one[i].split(';')[1].startswith(path_tree_one):
                lines_tree_one.append(i)
        for j in range(0, len(tree_two)):
            if tree_two[j].split(';')[1].startswith(path_tree_two):
                lines_tree_two.append(j)

        tree_one[tree_one_choice].split(';')[1].replace(path_tree_one, path_tree_two)
        tree_two[tree_two_choice].split(';')[1].replace(path_tree_two, path_tree_one)

        tree_one_src[tree_one_choice].split(';')[1].replace(path_tree_one, path_tree_two)
        tree_two_src[tree_two_choice].split(';')[1].replace(path_tree_two, path_tree_one)        

        new_tree_one = tree_one[:tree_one_choice]
        new_tree_two = tree_two[:tree_two_choice]

        new_tree_one_src = tree_one_src[:tree_one_choice]
        new_tree_two_src = tree_two_src[:tree_two_choice]

        for j in lines_tree_two:
            tmp_tree_two = tree_two[j].split(';')
            tmp_tree_two[0] = number_tree_one
            tmp_tree_two[1] = tmp_tree_two[1].replace(path_tree_two, path_tree_one, 1)
            new_tree_one.append(';'.join(tmp_tree_two))

            tmp_tree_two = tree_two_src[j].split(';')
            tmp_tree_two[0] = number_tree_one
            tmp_tree_two[1] = tmp_tree_two[1].replace(path_tree_two, path_tree_one, 1)
            new_tree_one_src.append(';'.join(tmp_tree_two))
        for k in lines_tree_one:
            tmp_tree_one = tree_one[k].split(';')
            tmp_tree_one[0] = number_tree_two
            tmp_tree_one[1] = tmp_tree_one[1].replace(path_tree_one, path_tree_two, 1)
            new_tree_two.append(';'.join(tmp_tree_one))

            tmp_tree_one = tree_one_src[k].split(';')
            tmp_tree_one[0] = number_tree_two
            tmp_tree_one[1] = tmp_tree_one[1].replace(path_tree_one, path_tree_two, 1)
            new_tree_two_src.append(';'.join(tmp_tree_one))

        new_tree_one.extend(tree_one[lines_tree_one[-1]+1:])
        new_tree_two.extend(tree_two[lines_tree_two[-1]+1:])

        new_tree_one_src.extend(tree_one_src[lines_tree_one[-1]+1:])
        new_tree_two_src.extend(tree_two_src[lines_tree_two[-1]+1:])

        ind1.individual_trees[part1] = new_tree_one
        ind2.individual_trees[part2] = new_tree_two
        ind1.first_source_tree[part1] = new_tree_one_src
        ind2.first_source_tree[part2] = new_tree_two_src

        ind1.predicate_inst.kb_source.extend(ind2.predicate_inst.kb_source)
        ind1.predicate_inst.new_kb_source.extend(ind2.predicate_inst.new_kb_source)
        ind1.predicate_inst.new_first_kb_source.extend(ind2.predicate_inst.new_first_kb_source)
        ind1.predicate_inst.kb_source = list(set(ind1.predicate_inst.kb_source))
        ind1.predicate_inst.new_kb_source = list(set(ind1.predicate_inst.new_kb_source))
        ind1.predicate_inst.new_first_kb_source = list(set(ind1.predicate_inst.new_first_kb_source))
        

        ind2.predicate_inst.kb_source.extend(ind1.predicate_inst.kb_source)
        ind2.predicate_inst.new_kb_source.extend(ind1.predicate_inst.new_kb_source)
        ind2.predicate_inst.new_first_kb_source.extend(ind1.predicate_inst.new_first_kb_source)
        ind2.predicate_inst.kb_source = list(set(ind2.predicate_inst.kb_source))
        ind2.predicate_inst.new_kb_source = list(set(ind2.predicate_inst.new_kb_source))
        ind2.predicate_inst.new_first_kb_source = list(set(ind2.predicate_inst.new_first_kb_source))

        return ind1, ind2

    def crossover_genes(self, pop_tree, elite_tree, src_pop_tree, src_elite_tree, positions, crossover_rate):
        new_tree = []
        new_src_tree = []

        base_individual_trees = len(pop_tree)
        smaller_tree = pop_tree
        if len(elite_tree) < len(pop_tree):
            base_individual_trees = len(elite_tree)
            smaller_tree = elite_tree

        for idx in range(0, base_individual_trees):
            if positions[idx] <= crossover_rate:
                new_tree.append(elite_tree[idx])
                new_src_tree.append(src_elite_tree[idx])
            else:
                new_tree.append(pop_tree[idx])
                new_src_tree.append(src_pop_tree[idx])


        if len(positions) > base_individual_trees:
            add = False
            for idx in range(base_individual_trees, len(positions)):
                if positions[idx] <= crossover_rate and smaller_tree == pop_tree:
                    new_tree.append(elite_tree[idx])
                    new_src_tree.append(src_elite_tree[idx])
                    add = True
                elif positions[idx] >= crossover_rate and smaller_tree == elite_tree:
                    new_tree.append(pop_tree[idx])
                    new_src_tree.append(src_pop_tree[idx])
                    add = True

            if add:
                for idx in range(base_individual_trees-1, len(new_tree)):
                    new_changed_tree = new_tree[idx].split(';')
                    new_changed_src_tree = new_src_tree[idx].split(';')
                    branch = new_tree[idx].split(';')[1]
                    for next_branch in range(idx+1, len(new_tree)):
                        if new_tree[next_branch].split(';')[1].startswith(branch):
                            change_branch = new_tree[next_branch].split(';')[1].split(f'{branch},')[1].split(',')[0]
                            if change_branch == 'false':
                                new_changed_tree[-1] = new_changed_tree[-1].replace(new_changed_tree[-1], 'true')
                                new_changed_src_tree[-1] = new_changed_src_tree[-1].replace(new_changed_src_tree[-1], 'true')
                            else:
                                new_changed_tree[-2] = new_changed_tree[-2].replace(new_changed_tree[-2], 'true')
                                new_changed_src_tree[-2] = new_changed_src_tree[-2].replace(new_changed_src_tree[-2], 'true')
                    new_tree[idx] = ";".join(new_changed_tree)
                    new_src_tree[idx] = ";".join(new_changed_src_tree)

        new_changed_tree = new_tree[-1].split(';')
        new_changed_src_tree = new_src_tree[-1].split(';')

        new_changed_tree[-2] = 'false'
        new_changed_tree[-1] = 'false'

        new_changed_src_tree[-2] = 'false'
        new_changed_src_tree[-1] = 'false'


        new_tree[-1] = ";".join(new_changed_tree)
        new_src_tree[-1] = ";".join(new_changed_src_tree)

        return new_tree, new_src_tree