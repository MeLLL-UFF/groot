import copy
from random import randint, random
import re


class Predicate:

    def __init__(self, kb_source, kb_target, target_pred):
        """
            Constructor

            Parameters
            ----------
            kb_source: list
            kb_target: list
            target_pred: list with tuples containing the predicates and theirs modes
                        example:
                        [('movie', '+,-'), ('director', '+'),...]
        """
        self.first_kb_source = kb_source
        self.kb_source = kb_source
        self.kb_target = kb_target
        self.target_pred = target_pred
        self.new_kb_source = []
        self.new_kb_target = []
        self.new_first_kb_source = []
        self.variables = []
        self.mapping_type = {}

    def check_tree(self, individual_tree, first_source_tree, ind):
        valid_tree = []
        first_source_pred = first_source_tree[0].split(';')[2].split(':-')[0].split('(')[0]
        first_target_pred = individual_tree[0].split(';')[2].split(':-')[0].split('(')[0]
        for line in range(0, len(individual_tree)):
            target_pred = individual_tree[line]
            source_pred = first_source_tree[line]
            res = individual_tree[line]

            if line == 0:
                #verifica os predicados junto ao target
                target = target_pred.split(';')[2].split(':-')[1].strip().split('),')
                source = source_pred.split(';')[2].split(':-')[1].strip().split('),')
                for idx in range(0, len(source)):
                    if not self.check_predicates(source[idx], target[idx], ind):
                        new_pred = self.get_modes(first_source_tree, 
                                                  first_source_tree[line].split(":- ")[1])
    
                        res = self.change_pred(first_source_pred, first_target_pred, 
                                                            first_source_tree[line], new_pred, ind)

                        new_target = res.split(';')[2].split(':-')[1].strip().split('),')[idx]
                        complete_source = self.get_complete_pred(source[idx], ind.predicate_inst.new_first_kb_source)
                        complete_target = self.get_complete_pred(new_target, ind.predicate_inst.new_kb_target)
                        ind.predicate_inst.define_mapping(complete_source, complete_target)

                    else:

                        tmp_res = res.split(';')
                        tmp_target = tmp_res[2].split(':-')
                        tmp_pred = tmp_target[1].strip().split('),')
                        tmp_pred[idx] = target[idx]
                        tmp_target[1] = '),'.join(tmp_pred)
                        tmp_res[2] = ':- '.join(tmp_target)
                        res = ";".join(tmp_res)
                valid_tree.append(res)
            else:
                target = target_pred.split(';')[2].split('),')
                source = source_pred.split(';')[2].split('),')
                
                res = individual_tree[line]
                for idx in range(0, len(source)):
                   
                    if not self.check_predicates(source[idx], target[idx], ind):
                        
                        new_pred = self.get_modes(first_source_tree, first_source_tree[line])
                    
                        res = self.change_pred(first_source_pred, first_target_pred, 
                                                            first_source_tree[line], new_pred, ind)
                        new_target = res.split(';')[2].split('),')[idx]

                        if "none" not in new_target:
                            complete_source = self.get_complete_pred(source[idx], ind.predicate_inst.new_first_kb_source)
                            complete_target = self.get_complete_pred(new_target, ind.predicate_inst.new_kb_target)
                            ind.predicate_inst.define_mapping(complete_source, complete_target)
                        
                    else:
                        tmp_res = res.split(';')
                        tmp = tmp_res[2].split('),')
                        tmp[idx] = target[idx]
                        tmp_res[2] = '),'.join(tmp)
                        res = ";".join(tmp_res)
                    
                if not 'none(none)' in res:
                    res = res.replace('(none)', 'none(none)')
                    
                valid_tree.append(res)
        ind = self.mapping_types(valid_tree, first_source_tree, ind)
        return valid_tree, ind

    def check_trees(self, ind):
        complete_source = self.get_complete_pred(ind.source, ind.predicate_inst.new_first_kb_source)
        complete_target = self.get_complete_pred(ind.target, ind.predicate_inst.new_kb_target)
        ind.predicate_inst.define_mapping(complete_source, complete_target)
        valid_trees = []
        for idx in range(0, len(ind.individual_trees)):
            valid_tree, ind = (self.check_tree(ind.individual_trees[idx], ind.first_source_tree[idx], ind))
            valid_trees.append(valid_tree)
            ind = self.mapping_types(ind.individual_trees[idx], ind.first_source_tree[idx], ind)
        return valid_trees

    def check_predicates(self, source_pred, target_pred, ind=None):
        has_target = False
        only_source = source_pred.split('(')[0].strip().lower()
        only_target = target_pred.split('(')[0].strip().lower()
        mapping_type = self.mapping_type
        if ind:
            mapping_type = ind.predicate_inst.mapping_type
        for pred in self.new_first_kb_source:
            if only_source == pred.split('(')[0].strip().lower():
                source_pred = pred
                break

        for pred in self.new_kb_target:
            if only_target == pred.split('(')[0].strip().lower():
                target_pred = pred
                has_target = True
                break
        if not has_target:
            return True
        source_types = source_pred.split('(')[1].split(')')[0].split(',')
        target_types = target_pred.split('(')[1].split(')')[0].split(',')

        for idx in range(0, len(source_types)):
            source_type = source_types[idx].strip()
            target_type = target_types[idx].strip()
            if source_type in list(mapping_type.keys()):
                if mapping_type[source_type] != target_type:
                    return False
        for idx in range(0, len(source_types)):
            if source_types[idx] not in list(mapping_type.keys()):
                self.mapping_type[source_types[idx]] = target_types[idx]
        return True

    def change_predicate(self, pred, new_info, var):
        """
            Define new types for the predicate and add additional infos (as tree number)

            Parameters
            ----------
            pred: string
            new_info: list
            var: string

            Returns
            ----------
            new_pred: string
        """
        #new_info é uma lista com o que se quer colocar entre pred e as variáveis
        #pred é do tipo: predicado(A)
        new_info = "".join(str(x) for x in new_info)
        return '{}{}{}'.format(pred.split('(')[0], new_info, var)

    def generate_new_preds(self):
        """ 
            Generate useful predicates from the knowledge base
        """
        self.new_kb_source = copy.deepcopy(self.kb_source)
        self.new_kb_target =copy.deepcopy(self.kb_target)

        for index in range(0, len(self.new_kb_source)):
            self.new_kb_source[index] = self.new_kb_source[index].replace('+', '')
            self.new_kb_source[index] = self.new_kb_source[index].replace('-', '')

        for index in range(0, len(self.new_kb_target)):
            self.new_kb_target[index] = self.new_kb_target[index].replace('+', '')
            self.new_kb_target[index] = self.new_kb_target[index].replace('-', '')

        if len(self.new_first_kb_source) == 0:
            self.new_first_kb_source =copy.deepcopy(self.first_kb_source)
            for index in range(0, len(self.first_kb_source)):
                self.new_first_kb_source[index] = self.new_first_kb_source[index].replace('+', '')
                self.new_first_kb_source[index] = self.new_first_kb_source[index].replace('-', '')

    def get_modes(self, individual_tree, pred_source):
        """
            Get the modes of the predicate

            Parameters
            ----------
            individual_tree: list
            pred_source: string

            Returns
            ----------
            modes: string
        """
        pred_source = pred_source.split('), ')
        predicate = []
        for pred in pred_source:
            if ')' not in pred:
                pred = f'{pred})'
            new_pred_source = f'{pred.split("(")[0]}('
            if not self.variables:
                self.variables.extend(("".join(re.findall('\((.*?)\)', individual_tree[0])[0])).split(', '))
            tmp_var = ("".join(re.findall('\((.*?)\)',pred))).split(',')
            self.variables = [i.strip() for i in self.variables]
            
            for var in tmp_var:
                if var.strip() in self.variables:
                    new_pred_source += '+,'
                else: 
                    new_pred_source += '-,'
                    self.variables.append(var)

            predicate.append(f'{new_pred_source})')
        return  ', '.join(predicate)

    def define_mapping(self, source_pred, target_pred):
        """
            Define the transfer between source and target types
            
            Parameters
            ----------
            source_pred: string
            target_pred: string
        """
        if source_pred and target_pred:
            source_pred = source_pred.split('(')[1].split(',')
            target_pred = target_pred.split('(')[1].split(',')
            for i in range(0, len(source_pred)):
                if source_pred[i].replace(').', '') not in list(self.mapping_type.keys()):
                    self.mapping_type[source_pred[i].replace(').', '')] = target_pred[i].replace(').', '')

    def mapping_types(self, individual_tree, first_source_tree, ind):
        first_source_pred = first_source_tree[0].split(';')[2].split(':-')[0].split('(')[0]
        first_target_pred = individual_tree[0].split(';')[2].split(':-')[0].split('(')[0]

        complete_source = self.get_complete_pred(first_source_pred, ind.predicate_inst.new_first_kb_source)
        complete_target = self.get_complete_pred(first_target_pred, ind.predicate_inst.new_kb_target)
        self.define_mapping(complete_source, complete_target)

        for line in range(0, len(individual_tree)):
            target_pred = individual_tree[line]
            source_pred = first_source_tree[line]
            
            if line == 0:
                #verifica os predicados junto ao target
                target = target_pred.split(';')[2].split(':-')[1].strip().split('),')
                source = source_pred.split(';')[2].split(':-')[1].strip().split('),')
            else:
                target = target_pred.split(';')[2].split('),')
                source = source_pred.split(';')[2].split('),')

            for idx in range(0, len(source)):
                complete_source = self.get_complete_pred(source[idx], ind.predicate_inst.new_first_kb_source)
                complete_target = self.get_complete_pred(target[idx], ind.predicate_inst.new_kb_target)
                if complete_source is not None and complete_target is not None:
                    ind.predicate_inst.define_mapping(complete_source, complete_target)
        return ind

    def _get_valid_map(self, complete_source, list_pos_target):
        """
            Get the possible mappings between types

            Parameters
            ----------
            complete_source: string
            list_pos_target: list

            Returns
            ----------
            index_target: list with indexes
        """
        pred_var = []
        list_vars = complete_source.split('(')[1].split(',')
        for i in list_vars:
            tmp = i.replace(').', '')
            if tmp in self.mapping_type.keys():
                pred_var.append(self.mapping_type[tmp])
            else:
                pred_var.append('')
        if len(f'({",".join(pred_var)}).') == 4:
            return list(range(0, len(list_pos_target)))
        else:
            pred_var = f'({",".join(pred_var)}).'

        index_target = []
        for var_target in range(0, len(list_pos_target)):
            if list_pos_target[var_target].endswith(pred_var.replace('(', '')):
                index_target.append(var_target)
            elif list_pos_target[var_target].split('(')[1].startswith(pred_var.replace(').', '').replace('(', '')):
                index_target.append(var_target)
        if len(index_target) == 0 and len(pred_var) == 3:
            pred_var = pred_var.replace('(', '').replace(').', '')
            for index in range(0, len(list_pos_target)):
                for i in pred_var:
                    if i in list_pos_target[index]:
                        return [index]

        return index_target

    def get_valid_predicates(self, source_pred, individual=None):
        """
            Get the possible predicates to be transfer with source_pred

            Parameters
            ----------
            source_pred: string

            Returns
            ----------
            complete_source_pred: string
            list_valid_index: list
            list_complete_valid_pred: list
        """
        occur_modes = ','.join([source_pred[occur.start()] 
                           for occur in re.finditer('[+\-]', source_pred)])
        valid_pred = []
        complete_valid_pred = []
        complete_source_pred = ''

        kb_source = self.kb_source
        new_kb_source = self.new_kb_source
        new_kb_target = self.new_kb_target
        if individual:
            kb_source = individual.predicate_inst.first_kb_source
            new_kb_source = individual.predicate_inst.new_first_kb_source
            new_kb_target = individual.predicate_inst.new_kb_target
        
        new_source_pred = source_pred.split('(')[0].strip()
        if ';' in new_source_pred:
            new_source_pred = new_source_pred.split(';')[2]

        for pred in kb_source:
            if new_source_pred in pred:
                source_pred = pred
                break
        occur_modes = ','.join([source_pred[occur.start()] 
                           for occur in re.finditer('[+\-]', source_pred)])

        for pred in new_kb_source:
            if new_source_pred in pred:
                complete_source_pred = pred
                break

        if new_source_pred == 'none':
            return '', [], []

        if complete_source_pred == '':
            print(source_pred)
            print(new_source_pred)
            print("KB: ", new_kb_source)
            print(individual.individual_trees)

        for pred, mode in self.target_pred:
            if mode == occur_modes: valid_pred.append(f'{pred}({mode})')

        for target in valid_pred:
            for pred in new_kb_target:
                if target.split('(')[0].strip() in pred:
                    complete_valid_pred.append(pred)
                    break
        valid_index = self._get_valid_map(complete_source_pred, complete_valid_pred)
        return complete_source_pred, [valid_pred[index] for index in valid_index], [complete_valid_pred[index] for index in valid_index]

    def get_complete_pred(self, pred, kb):
        """
            Get complete predicate from kb, with the types

            Parameters
            ----------
            pred: string
            kb: list

            Returns
            ----------
            complete_predicate: string
        """
        # pred é da forma: pred(A, B) ou pred
        for i in kb:
            if pred.split('(')[0].strip().lower() == i.split('(')[0].strip().lower():
                return i

    def new_pred(self, source_pred, target_pred):
        """
            Returns the target predicate with the source types

            Returns
            ----------
            predicate: string
        """
        return f'{target_pred.split("(")[0]}({source_pred.split("(")[1]}'

    def change_pred(self, source, target, pred, source_pred, individual=None):
        """
            Change the predicate pred with the random predicates from source_pred
            If the predicate is the main predicate (source), it will be changed with target

            Parameters
            ----------
            source: string
            target: string
            pred: string
            source_pred: string

            Returns
            ----------
            predicate: string
        """
        new_kb_source = self.new_kb_source
        new_kb_target = self.new_kb_target
        if individual:
            new_kb_source = individual.predicate_inst.new_first_kb_source
            new_kb_target = individual.predicate_inst.new_kb_target

        split_pred = pred.split(";")
        has_target = len(split_pred[2].split(":-")) > 1
        if has_target: 
            target_pred = split_pred[2].split(":- ")[1]
        else: target_pred = split_pred[2]

        source_pred = source_pred.split('), ')
        final_pred = []
        qtd_preds = 0
        for pred in source_pred:
            complete_source, valid_preds, complete_valid = self.get_valid_predicates(pred, individual)
            if not len(valid_preds):
                if has_target: 
                    pp = split_pred[2].split(':-')[1].split(')')
                    types = f"({pp[qtd_preds].strip().split('(')[1]})"
                else:
                    pp = split_pred[2].split(')')
                    types = f"({pp[qtd_preds].strip().split('(')[1]})"
                final_pred.append(f'none{types}')
                continue
            index_choice = randint(0, len(valid_preds)-1)
            if has_target:
                pp = split_pred[2].split(':-')[1].split(')')
                types = f"({pp[qtd_preds].strip().split('(')[1]})"
                final_pred.append(f"{complete_valid[index_choice].split('(')[0]}{types}")

            else:
                pp = split_pred[2].split(')')
                types = f"({pp[qtd_preds].strip().split('(')[1]})"
                final_pred.append(f"{complete_valid[index_choice].split('(')[0]}{types}")
            qtd_preds += 1
        if has_target: 
            split_pred[2] =  f"{target}({split_pred[2].split(':-')[0].split('(')[1].strip()} :- {', '.join(final_pred)}"
        else: split_pred[2] = ', '.join(final_pred)
        split_pred[2] = f"{split_pred[2]}."
        return ";".join(split_pred)
