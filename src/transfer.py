import string


class Transfer:

    def __init__(self, predicate_instance, target):
        self.predicate_inst = predicate_instance #<---- Predicate(...)
        self.transfer = []
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
        # linha é do tipo: 0;;principal :- outro_predicado ...
        no_var_prc, no_var_sec = self._count_vars(source_pred)
        var_prc = self._generate_new_pred(no_var_prc)
        var_sec = self._generate_new_pred(no_var_sec)
        #principal
        self.mapping_transfer(source_pred.split(";")[2].split(" :-")[0], 
                              self.target, 
                              tree_number, 0)
        new_pred_prc = self.predicate_inst.change_predicate(source_pred.split(";")[2].split(":-")[0], 
                                                            [tree_number, 0], 
                                                            var_prc)
        #predicado que está junto ao target
        self.mapping_transfer(source_pred.split(";")[2].split(":- ")[1], 
                              target_pred.split(";")[2].split(":- ")[1],
                              tree_number, 0)

        new_pred_sec = self.predicate_inst.change_predicate(source_pred.split(";")[2].split(":-")[1], 
                                                            [tree_number, 0], 
                                                            var_sec)


        return new_pred_prc, new_pred_sec

    def mapping_transfer(self, source_pred, target_pred, tree_number, index_node):
        #source é do tipo: pred(A) e target é pred(A)
        new_source_pred = source_pred.split('(')[0].rstrip('0123456789')
        new_target_pred = target_pred.split('(')[0].rstrip('0123456789')
        no_var = self._count_vars(source_pred)
        var_pred = self._generate_new_pred(no_var)
                                                    
        for pred in self.predicate_inst.new_bk_source:
            if new_source_pred in pred: 
                real_predicate = '({}'.format(pred.split('(')[1])
                self.transfer.append('source: {}'.format(self.predicate_inst.change_predicate(pred, 
                                                                                              [tree_number, index_node], 
                                                                                              real_predicate)))
                break
        for pred in self.predicate_inst.new_bk_target:
            if new_target_pred in pred: 
                self.transfer.append('target: {}'.format(pred))
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
        modf_src_tree = []
        for line in range(0, len(individual_tree)):
            pred_target = individual_tree[line]
            pred_source = source_tree[line]
            if line == 0:
                new_pred_prc, new_pred_sec = self.mapping_principal_pred(pred_source, 
                                                                         pred_target, 
                                                                         tree_number)

                pred_source = pred_source.replace(pred_source.split(";")[2].split(":-")[0],
                                                   new_pred_prc)

                modf_src_tree.append(pred_source.replace(pred_source.split(";")[2].split(":-")[1],
                                                         new_pred_sec))
            else:
                self.mapping_transfer(pred_source.split(";")[2], 
                                      pred_target.split(";")[2],
                                      tree_number, 
                                      line)
                no_var = self._count_vars(pred_source)
                var_pred = self._generate_new_pred(no_var)
                new_src = self.predicate_inst.change_predicate(pred_source.split(";")[2], 
                                                               [tree_number, line], 
                                                               var_pred)
                modf_src_tree.append(pred_source.replace(pred_source.split(";")[2],
                                                         new_src))
        return modf_src_tree

    def mapping_all_trees(self, individual_trees, source_trees):
        self.predicate_inst.generate_new_preds()
        self.transfer = []
        modified_src_tree = []
        for i in range(0, len(individual_trees)):
            modified_src_tree.append(self.mapping_tree(individual_trees[i],
                                                       source_trees[i], 
                                                       i))
        # self.include_parameters()
        return modified_src_tree
