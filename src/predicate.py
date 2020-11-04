import copy
from random import randint, random
import re


class Predicate:

    def __init__(self, bk_source, bk_target, pred_target):
        self.bk_source = bk_source
        self.bk_target = bk_target
        self.pred_target = pred_target
        self.new_bk_source = []
        self.new_bk_target = []
        self.variables = []

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

    def change_predicate(self, pred, new_info, var):
        #new_info é uma lista com o que se quer colocar entre pred e as variáveis
        #pred é do tipo: predicado(A)
        new_info = "".join(str(x) for x in new_info)
        return '{}{}{}'.format(pred.split('(')[0], new_info, var)

    def generate_new_preds(self):
        self.new_bk_source = copy.deepcopy(self.bk_source)
        self.new_bk_target =copy.deepcopy(self.bk_target)

        for index in range(0, len(self.new_bk_source)):
            self.new_bk_source[index] = self.new_bk_source[index].replace('+', '')
            self.new_bk_source[index] = self.new_bk_source[index].replace('-', '')

        for index in range(0, len(self.new_bk_target)):
            self.new_bk_target[index] = self.new_bk_target[index].replace('+', '')
            self.new_bk_target[index] = self.new_bk_target[index].replace('-', '')

    def get_modes(self, individual_tree, pred_source):
        #LEVA EM CONSIDERAÇAO QUE SÓ HÁ UM PREDICADO EM CADA NÓ
        new_pred_source = f'{pred_source.split("(")[0]}('
        if not self.variables:
            self.variables.extend(("".join(re.findall('\((.*?)\)', individual_tree[0])[0])).split(', '))
        tmp_var = ("".join(re.findall('\((.*?)\)',pred_source))).split(',')
        for var in tmp_var:
            if var in self.variables:
                new_pred_source += '+,'
            else: 
                new_pred_source += '-,'
                self.variables.append(var)
        return f'{new_pred_source})'

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

    def new_pred(self, source_pred, target_pred):
        return f'{target_pred.split("(")[0]}({source_pred.split("(")[1]}'

    def change_pred(self, pred, pred_source):
        split_pred = pred.split(";")
        has_target = len(split_pred[2].split(":-")) > 1
        if has_target: target_pred = split_pred[2].split(":- ")[1]
        else: target_pred = split_pred[2]
        var_pred = '({}'.format(target_pred.split('(')[1])
        valid_preds = self.get_valid_predicates(pred_source)
        # print(valid_preds)
        index_choice = randint(0, len(valid_preds)-1)
        target_pred = self.new_pred(target_pred, valid_preds[index_choice])
        if has_target: 
            split_pred[2] =  f"{split_pred[2].split(':-')[0]} :- {target_pred}"
        else: split_pred[2] = target_pred
        return ";".join(split_pred)
