import string


class Transfer:

    def __init__(self, predicate_instance, target):
        self.predicate_inst = predicate_instance #<---- Predicate(...)
        self.transfer = []
        self.transfer_variables = {}
        self.target = target

    def _generate_new_pred(self, no_var):
        return '({})'.format(",".join(list(string.ascii_uppercase)[0:no_var]))

    def _count_vars(self, pred):
        if len(pred.split(":-")) > 1:
            #retorna o principal e o que vem a seguir
            return (len(pred.split(";")[2].split(":-")[0].split('(')[1].split(',')),
                   len(pred.split(";")[2].split(":-")[1].split('(')[1].split(',')))
        else:
            return len(pred.split('(')[1].split(','))

    def mapping_principal_pred(self, source_pred, target_pred, tree_number):
        # linha é do tipo: 0;;principal :- outro_predicado, outro predicado ...
        new_pred_sec = []
        source_pred = source_pred.split('), ')
        target_pred = target_pred.split('), ')
        # print(source_pred[0])
        no_var_prc, no_var_sec = self._count_vars(source_pred[0])
        var_prc = self._generate_new_pred(no_var_prc)
        var_sec = f"({source_pred[0].split(';')[2].split(':-')[1].split('(')[1].split(').')[0].split(')')[0]})"#self._generate_new_pred(no_var_sec)
        
        # print("PRINCIPAL AND SECOND: ")
        # print(var_prc, var_sec)
        # print("--------------------------")
        #principal
        self.mapping_transfer(source_pred[0].split(";")[2].split(" :-")[0], 
                              self.target, 
                              tree_number, 0)
        new_pred_prc = self.predicate_inst.change_predicate(source_pred[0].split(";")[2].split(":-")[0], 
                                                            [tree_number, 0], 
                                                            var_prc)
        #predicado que está junto ao target
        self.mapping_transfer(source_pred[0].split(";")[2].split(":- ")[1], 
                              target_pred[0].split(";")[2].split(":- ")[1],
                              tree_number, 0)

        new_pred_sec.append(self.predicate_inst.change_predicate(source_pred[0].split(";")[2].split(":-")[1], 
                                                            [tree_number, 0], 
                                                            var_sec))
        for index in range(1, len(source_pred)):
          #pego os outros predicados
          self.mapping_transfer(source_pred[index].split(";")[0], 
                                target_pred[index].split(";")[0], 
                                tree_number, 0)

          no_var = self._count_vars(source_pred[index])
          # var_pred = self._generate_new_pred(no_var)
          var_pred = f"({source_pred[index].split('(')[1].split(').')[0].split(')')[0]})"
          # print(var_pred)
          new_pred_sec.append(self.predicate_inst.change_predicate(source_pred[index].split(";")[0], 
                                                        [tree_number, 0], 
                                                        var_pred))
        return new_pred_prc, ", ".join(new_pred_sec)

    def _get_complete_predicate(self, predicate, bk):
        for pred in bk:
            if predicate in pred:
                return pred

    def def_transfer(self, target, source):
        target = target.split('(')[1].split(',')
        source = source.split('(')[1].split(',')
        for i in range(0, len(target)):
            self.transfer_variables[source[i].replace(').', '')] = target[i].replace(').', '')
        # print(self.transfer_variables)

    def mapping_variables(self, pred_source, pred_target):
        target = pred_target.split('(')[1].split(',')
        source = pred_source.split('(')[1].split(',')
        res = []
        new_mapping = True
        tmp_dict = {}
        for i in range(0, len(target)):
            tmp = source[i].replace(').', '')
            if tmp in self.transfer_variables.keys():
                new_mapping = False
                break
            else: 
                tmp_dict[tmp] = target[i].replace(').', '')
        if new_mapping:
            self.transfer_variables.update(tmp_dict)
            return True
        else:
            for i in source:
                try: 
                    # print("TESTE: ", i)
                    res.append(self.transfer_variables[i.replace(').', '')])
                except KeyError:
                    # print("DEU ERRO: ", i)
                    res.append('')
            return f'({",".join(res)}).'

    def mapping_transfer(self, source_pred, target_pred, tree_number, index_node):
        #source é do tipo: pred(A) e target é pred(A)
        source_pred = source_pred.split('), ')
        target_pred = target_pred.split('), ')
        # print("SOURCE E TARGET: ")
        # print(source_pred, target_pred)
        # print("+++++++++++++++++++++++")
        for index in range(0, len(source_pred)):
            new_source_pred = source_pred[index].split('(')[0].rstrip('0123456789')
            new_target_pred = target_pred[index].split('(')[0].rstrip('0123456789')
            no_var = self._count_vars(source_pred[index])
            var_pred = self._generate_new_pred(no_var)
            # var_pred = f"({source_pred[index].split('(')[1].split(').')[0].split(')')[0]})"
            # print("VAR PRED: ", var_pred)
            # print("------------------------")
            for pred in self.predicate_inst.new_bk_source:
                if new_source_pred in pred: 
                    real_predicate = '({}'.format(pred.split('(')[1])
                    source = 'source: {}'.format(self.predicate_inst.change_predicate(pred, 
                                                                                      [tree_number, index_node], 
                                                                                      real_predicate))
                    if source not in self.transfer:
                        self.transfer.append(source)
                    break
            for pred in self.predicate_inst.new_bk_target:
                if new_target_pred in pred: 
                    target = 'target: {}'.format(pred)
                    if target not in self.transfer:
                        self.transfer.append(target)
                    break

            setMap = 'setMap: {}={}{}.'.format(self.predicate_inst.change_predicate(new_source_pred, 
                                               [tree_number, index_node], var_pred),
                                                new_target_pred, 
                                                var_pred)
            if setMap not in self.transfer:
                self.transfer.append(setMap)
        
    def include_parameters(self):
        self.transfer.append('setParam: searchArgPermutation=true.')
        self.transfer.append('setParam: searchEmpty=false.')
        self.transfer.append('setParam: allowSameTargetMap=false.')

    def mapping_tree(self, individual_tree, source_tree, tree_number):
        # print("ESTOU AQUI: ")
        # print(individual_tree)
        modf_src_tree = []
        for line in range(0, len(individual_tree)):
            individual_tree[line] = str(tree_number) + individual_tree[line][1:]
            source_tree[line] = str(tree_number) + source_tree[line][1:]
            # print(source_tree[line])
            pred_target = individual_tree[line]
            pred_source = source_tree[line]
            if line == 0:
                new_pred_prc, new_pred_sec = self.mapping_principal_pred(pred_source, 
                                                                         pred_target, 
                                                                         tree_number)

                pred_source = pred_source.replace(pred_source.split(";")[2].split(":-")[0],
                                                   new_pred_prc)
                # print("PRINCIPAL: ")
                # print(pred_source)
                # print(new_pred_sec)
                # print("-------------------")

                modf_src_tree.append(pred_source.replace(pred_source.split(";")[2].split(":-")[1],
                                                         new_pred_sec))
            else:
                self.mapping_transfer(pred_source.split(";")[2], 
                                      pred_target.split(";")[2],
                                      tree_number, 
                                      line)

                pred_source = pred_source.split(';')
                predicates = pred_source[2].split('), ')
                new_pred = []
                for index in range(0, len(predicates)):
                    no_var = self._count_vars(predicates[index])
                    var_pred = f"({predicates[index].split('(')[1].split(').')[0].split(')')[0]}"
                    new_src = self.predicate_inst.change_predicate(predicates[index], 
                                                             [tree_number, line], 
                                                             var_pred)
                    # print(new_src, predicates[index])

                    new_src = f'{new_src.split("(")[0]}({predicates[index].split("(")[1]}'
                    if ')' not in new_src:
                        new_src = f'{new_src})'
                    
                    new_pred.append(new_src)
                pred_source[2] = pred_source[2].replace(pred_source[2],f'{", ".join(new_pred)}')
                modf_src_tree.append(";".join(pred_source))
        return modf_src_tree

    def mapping_all_trees(self, individual_trees, source_trees):
        self.predicate_inst.generate_new_preds()
        self.transfer = []
        modified_src_tree = []
        # print('NUMERO DAS ARVORES: ')
        for i in range(0, len(individual_trees)):
            # print(i)
            modified_src_tree.append(self.mapping_tree(individual_trees[i],
                                                       source_trees[i], 
                                                       i))
        self.include_parameters()
        # print('----------------------')
        return modified_src_tree
