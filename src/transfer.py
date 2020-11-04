import string


class Transfer:

    def __init__(self, predicate_instance, target):
        self.predicate_inst = predicate_instance #<---- Predicate(...)
        self.transfer = []
        self.target = target

    def mapping_transfer(self, source_pred, target_pred, tree_number, index_node, with_number=True):
        #source é do tipo: pred(A) e target é pred(A)
        new_source_pred = source_pred.split('(')[0].rstrip('0123456789')
        new_target_pred = target_pred.split('(')[0].rstrip('0123456789')
        no_var = len(source_pred.split('(')[1].split(','))
        var_pred = '({})'.format(",".join(list(string.ascii_uppercase)[0:no_var]))
                                                    
        if with_number:
            # print(self.predicate_inst.new_bk_source)
            for pred in self.predicate_inst.new_bk_source:
                # 
                # print(new_source_pred in pred, 'source: {}.'.format(pred) not in self.transfer)
                if new_source_pred in pred and 'source: {}.'.format(pred) not in self.transfer: 
                    self.transfer.append('source: {}.'.format(self.predicate_inst.change_predicate(new_source_pred, 
                                                                [tree_number, index_node], var_pred)))
                    break
            for pred in self.predicate_inst.new_bk_target:
                if new_target_pred in pred and 'target: {}.'.format(pred) not in self.transfer:
                    self.transfer.append('target: {}'.format(pred))
                    break
            if 'setMap: {}={}{}.'.format(self.predicate_inst.change_predicate(new_source_pred, [tree_number, index_node], var_pred),
                                                new_target_pred, var_pred) not in self.transfer:
                self.transfer.append('setMap: {}={}{}.'.format(self.predicate_inst.change_predicate(new_source_pred, 
                                                                [tree_number, index_node], var_pred),
                                                                new_target_pred, var_pred))
        else:
            for pred in self.predicate_inst.new_bk_source:
                if new_source_pred in pred and 'source: {}.'.format(pred) not in self.transfer: 
                    self.transfer.append('source: {}.'.format(source_pred))
                    break
            for pred in self.predicate_inst.new_bk_target:
                if new_target_pred in pred and 'target: {}'.format(pred) not in self.transfer:
                    self.transfer.append('target: {}'.format(pred))
                    break
            if 'setMap: {}={}{}.'.format(source_pred, new_target_pred, var_pred) not in self.transfer:
                self.transfer.append('setMap: {}={}{}.'.format(source_pred, new_target_pred, var_pred))

    def mapping_tree(self, individual_tree, source_tree, tree_number):
        modf_src_tree = []
        for line in range(0, len(individual_tree)):
            pred_target = individual_tree[line]
            pred_source = source_tree[line]
            if line == 0:
                no_var = len(pred_source.split(";")[2].split(":-")[0].split('(')[1].split(','))
                var_pred = '({})'.format(",".join(list(string.ascii_uppercase)[0:no_var]))
                #target
                self.mapping_transfer(pred_source.split(";")[2].split(" :-")[0], self.target, 
                                      tree_number, line, True)
                new_pred_src = self.predicate_inst.change_predicate(pred_source.split(";")[2].split(":-")[0], 
                                                        [tree_number, line], var_pred)
                pred_source = pred_source.replace(pred_source.split(";")[2].split(":-")[0],
                                                         new_pred_src)
                #predicado que está junto ao target
                self.mapping_transfer(pred_source.split(";")[2].split(":- ")[1], 
                                      pred_target.split(";")[2].split(":- ")[1],
                                      tree_number, line)
                no_var = len(pred_source.split(";")[2].split(":-")[1].split('(')[1].split(','))
                var_pred = '({})'.format(",".join(list(string.ascii_uppercase)[0:no_var]))
                new_pred_src = self.predicate_inst.change_predicate(pred_source.split(";")[2].split(":-")[1], 
                                                        [tree_number, line], var_pred)

                modf_src_tree.append(pred_source.replace(pred_source.split(";")[2].split(":-")[1],
                                                         new_pred_src))
            else:
                self.mapping_transfer(pred_source.split(";")[2], 
                                      pred_target.split(";")[2],
                                      tree_number, line)
                no_var = len(pred_source.split('(')[1].split(','))
                var_pred = '({})'.format(",".join(list(string.ascii_uppercase)[0:no_var]))
                new_src = self.predicate_inst.change_predicate(pred_source.split(";")[2], 
                                                [tree_number, line], var_pred)
                modf_src_tree.append(pred_source.replace(pred_source.split(";")[2],
                                                         new_src))
        return modf_src_tree

    def mapping_all_trees(self, individual_trees, source_trees):
        self.predicate_inst.generate_new_preds()
        self.transfer = []
        modified_src_tree = []
        for i in range(0, len(individual_trees)):
            modified_src_tree.append(self.mapping_tree(individual_trees[i],
                                                            source_trees[i], i))
        return modified_src_tree
