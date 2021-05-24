import string


class Transfer:

    def __init__(self, predicate_instance, target):
        """
            Constructor
            
            Parameters
            ----------
            predicate_instance: Predicate instance
            target: string
        """
        self.predicate_inst = predicate_instance #<---- Predicate(...)
        self.transfer = []
        self.transfer_variables = {}
        self.target = target

    def _generate_new_pred(self, no_var, has_permutation=False):
        """
            Generate predicate according to the arity

            Parameters
            ----------
            no_var: int

            Returns
            ----------
            predicate: string
        """
        if has_permutation:
            return '({})'.format(",".join(list(string.ascii_uppercase)[0:2][::-1]))
        return '({})'.format(",".join(list(string.ascii_uppercase)[0:no_var]))

    def _count_vars(self, pred):
        """
            Counting the arity of predicate

            Parameters
            ----------
            pred: string

            Returns
            ----------
            arity: int
        """
        if len(pred.split(":-")) > 1:
            #retorna o principal e o que vem a seguir
            return (len(pred.split(";")[2].split(":-")[0].split('(')[1].split(',')),
                   len(pred.split(";")[2].split(":-")[1].split('(')[1].split(',')))
        else:
            return len(pred.split('(')[1].split(','))

    def mapping_principal_pred(self, source_pred, target_pred, tree_number, transfer, ind):
        """
            Mapping the transfer for the first predicate in the tree
            The tree structure has, at first, the target predicate and the immediate predicate
            This method makes the mapping from the source to target predicates of both predicates
            
            Parameters
            ----------
            source_pred: string
            target_pred: string
            tree_number: int

            Returns
            ----------
            new_pred_prc: string
            new_pred_sec: string
        """
        # linha é do tipo: 0;;principal :- outro_predicado, outro predicado ...
        new_pred_sec = []
        source_pred = source_pred.split('), ')
        target_pred = target_pred.split('), ')
        no_var_prc, no_var_sec = self._count_vars(source_pred[0])
        var_prc = self._generate_new_pred(no_var_prc)
        var_sec = f"({source_pred[0].split(';')[2].split(':-')[1].split('(')[1].split(').')[0].split(')')[0]})"#self._generate_new_pred(no_var_sec)
    
        #principal
        self.mapping_transfer(source_pred[0].split(";")[2].split(" :-")[0], 
                              self.target, 
                              tree_number, 0, transfer)
        new_pred_prc = self.predicate_inst.change_predicate(source_pred[0].split(";")[2].split(":-")[0], 
                                                            [tree_number, 0], 
                                                            var_prc)
    
        res = ind.predicate_inst.check_predicates(source_pred[0].split(";")[2].split(" :-")[0], self.target, ind)
        # if not res:
        #     print(f'{ind.predicate_inst.mapping_type}')
        #     print('=================================')
        #     print(f'{source_pred[0].split(";")[2].split(" :-")[0]} & {self.target} == {res}')
        #     print('=================================')
        #     print(target_pred, source_pred)
            # exit()
        
        res_new_preds = ind.predicate_inst.verify_permutation(source_pred[0].split(";")[2].split(":- ")[1], target_pred[0].split(";")[2].split(":- ")[1], ind)
       
        source_pred[0] = source_pred[0].replace(source_pred[0].split(";")[2].split(":- ")[1], res_new_preds[0])
        target_pred[0] = target_pred[0].replace(target_pred[0].split(";")[2].split(":- ")[1], res_new_preds[1])
        
        ## ====== predicado que está junto ao target ====== ##
        self.mapping_transfer(source_pred[0].split(";")[2].split(":- ")[1], 
                              target_pred[0].split(";")[2].split(":- ")[1],
                              tree_number, 1, transfer, res_new_preds[2])

        
        var_sec = f"({source_pred[0].split(';')[2].split(':-')[1].split('(')[1].split(').')[0].split(')')[0]})"
        new_pred_sec.append(self.predicate_inst.change_predicate(source_pred[0].split(";")[2].split(":-")[1], 
                                                            [tree_number, 1], 
                                                            var_sec))

        res = ind.predicate_inst.check_predicates(source_pred[0].split(";")[2].split(" :-")[1], 
                                                   target_pred[0].split(";")[2].split(":- ")[1],
                                                   ind)


        # if not res:
        #     print(f'{ind.predicate_inst.mapping_type}')
        #     print('=================================')
        #     print(f'{source_pred[0].split(";")[2].split(" :-")[1]} & {target_pred[0].split(";")[2].split(":- ")[1]} == {res}')
        #     print('=================================')
        #     print(target_pred, source_pred)
        #     print('=================================')
        #     print(ind.predicate_inst.mapping_type)
            # exit()
        for index in range(1, len(source_pred)):
            #pego os outros predicados
            res_new_preds = ind.predicate_inst.verify_permutation(source_pred[index].split(";")[0], target_pred[index].split(";")[0], ind)
            
           
            source_pred[index] = source_pred[index].replace(source_pred[index].split(";")[0], res_new_preds[0])
            target_pred[index] = target_pred[index].replace(target_pred[index].split(";")[0], res_new_preds[1])
    
            self.mapping_transfer(source_pred[index].split(";")[0], 
                                target_pred[index].split(";")[0], 
                                tree_number, index, transfer, res_new_preds[2])

        
            res = ind.predicate_inst.check_predicates(source_pred[index].split(";")[0], 
                                                   target_pred[index].split(";")[0],
                                                   ind)

            # if not res:
            #     print(f'{ind.predicate_inst.mapping_type}')
            #     print('=================================')
            #     print(f'{source_pred[index].split(";")[0]} & {target_pred[index].split(";")[0]} == {res}')
            #     print('=================================')
            #     print(target_pred, source_pred)
                # exit()
            no_var = self._count_vars(source_pred[index])
            var_pred = f"({source_pred[index].split('(')[1].split(').')[0].split(')')[0]})"
            new_pred_sec.append(self.predicate_inst.change_predicate(source_pred[index].split(";")[0], 
                                                        [tree_number, index], 
                                                        var_pred))

        return new_pred_prc, ", ".join(new_pred_sec), transfer

    def mapping_transfer(self, source_pred, target_pred, tree_number, index_node, transfer, has_permutation=False):
        """
            Mapping the transfer of the predicate in the tree

            Parameters
            ----------
            source_pred: string
            target_pred: string
            tree_number: int
            index_node: int
        """
        #source é do tipo: pred(A) e target é pred(A)
        if 'none' in target_pred:
            return transfer
        source_pred = source_pred.split('), ')
        target_pred = target_pred.split('), ')
        for index in range(0, len(target_pred)):
            if target_pred[index] == '':
                return transfer
            new_source_pred = source_pred[index].split('(')[0].rstrip('0123456789').lower().strip()
            new_target_pred = target_pred[index].split('(')[0].rstrip('0123456789').lower().strip()
            if len(new_target_pred) == 0:
                continue
            no_var = self._count_vars(source_pred[index])
            var_pred = self._generate_new_pred(no_var)
            for pred in self.predicate_inst.new_first_kb_source:
                if new_source_pred == pred.split('(')[0].lower().strip(): 
                    real_predicate = f"({pred.split('(')[1].replace('+', '').replace('-', '').replace('`', '')}"
                    source = 'source: {}'.format(self.predicate_inst.change_predicate(pred, 
                                                                                      [tree_number, index_node], 
                                                                                      real_predicate))
                    if source not in transfer:
                        transfer.append(source)
                    break
            for pred in self.predicate_inst.new_first_kb_source:
                if new_source_pred == pred.split('(')[0].lower().strip():  
                    real_predicate = f"({pred.split('(')[1].replace('+', '').replace('-', '').replace('`', '')}"
                    source = 'source: {}'.format(self.predicate_inst.change_predicate(pred, 
                                                                                      [tree_number, index_node], 
                                                                                      real_predicate))
                    if source not in transfer:
                        transfer.append(source)
                    break

            for pred in self.predicate_inst.new_kb_target:
                if new_target_pred == pred.split('(')[0].lower().strip(): 
                    if has_permutation:
                        pred_args = f"{pred.split('(')[0].lower().strip()}({','.join(pred.split('(')[1].lower().strip().replace(').', '').split(',')[::-1])})."
                        target = 'target: {}'.format(pred_args)
                    else:
                        target = 'target: {}'.format(pred)
                    # if target not in transfer:
                    transfer.append(target)
                    break


            setMap = 'setMap: {}={}{}.'.format(self.predicate_inst.change_predicate(new_source_pred, 
                                               [tree_number, index_node], var_pred),
                                                new_target_pred, 
                                                var_pred)
            if setMap not in transfer:
                transfer.append(setMap)
        return transfer
        
    def include_parameters(self, transfer):
        """
            Include additional parameters to the transfer
        """
        transfer.append('setParam: searchArgPermutation=true.')
        transfer.append('setParam: searchEmpty=false.')
        transfer.append('setParam: allowSameTargetMap=false.')
        return transfer

    def mapping_tree(self, individual_tree, source_tree, tree_number, transfer, ind):
        """
            Mapping the transfer between the source and target tree
            It also returns the tree modified with the predicates which will be used in the transfer

            Parameters
            ----------
            individual_tree: list
            source_tree: list
            tree_number: int

            Returns
            ----------
            modf_src_tree: list
        """
        modf_src_tree = []
        for line in range(0, len(individual_tree)):
            individual_tree[line] = str(tree_number) + individual_tree[line][1:]
            source_tree[line] = str(tree_number) + source_tree[line][1:]
            pred_target = individual_tree[line]
            pred_source = source_tree[line]
            if line == 0:
                new_pred_prc, new_pred_sec, transfer = self.mapping_principal_pred(pred_source, 
                                                                                     pred_target, 
                                                                                     tree_number, transfer,
                                                                                     ind)
                pred_source = pred_source.replace(pred_source.split(";")[2].split(":-")[0],
                                                   new_pred_prc)
                modf_src_tree.append(pred_source.replace(pred_source.split(";")[2].split(":-")[1],
                                                         new_pred_sec))
            else:
                pred_source = pred_source.split(';')
                predicates = pred_source[2].split('), ')
                pred_target = pred_target.split(';')[2].split('), ')


                new_pred = []
                start = predicates
                if len(pred_target) < len(predicates): start = pred_target
                for index in range(0, len(start)):
                    if predicates[index].split('(')[1] == pred_target[index].split('(')[1]:

                        res_new_preds = ind.predicate_inst.verify_permutation(predicates[index], pred_target[index], ind)
                        
                        predicates[index] = predicates[index].replace(predicates[index], res_new_preds[0])
                        pred_target[index] = pred_target[index].replace(pred_target[index], res_new_preds[1])
                                    
                        res = ind.predicate_inst.check_predicates(predicates[index], pred_target[index], ind)
                        # if not res:
                        #     print(f'{ind.predicate_inst.mapping_type}')
                        #     print('=================================')
                        #     print(f'{predicates[index]} & {pred_target[index]} == {res}')
                        #     print('=================================')
                        #     print(predicates, pred_source)
                            # exit()
                        tranfer = self.mapping_transfer(predicates[index], 
                                                          pred_target[index],
                                                          tree_number, 
                                                          line+index+1, 
                                                          transfer, res_new_preds[2])
                        no_var = self._count_vars(predicates[index])
                        var_pred = f"({predicates[index].split('(')[1].split(').')[0].split(')')[0]}"
                        new_src = self.predicate_inst.change_predicate(predicates[index], 
                                                                 [tree_number, line+index+1], 
                                                                 var_pred)

                        new_src = f'{new_src.split("(")[0]}({predicates[index].split("(")[1]}'
                        if ')' not in new_src:
                            new_src = f'{new_src})'
                        
                        new_pred.append(new_src)
                        
                pred_source[2] = pred_source[2].replace(pred_source[2],f'{", ".join(new_pred)}')
                if '.' not in pred_source[2]: pred_source[2] = f'{pred_source[2]}.'
                modf_src_tree.append(";".join(pred_source))
        return modf_src_tree, transfer

    def mapping_all_trees(self, individual_trees, source_trees, ind):
        """
            Make the transfer mapping of all trees

            Parameters
            ----------
            individual_trees: list of lists 
            source_trees: list of lists

            Returns
            ----------
            modified_src_tree: list of lists
            transfer: list
        """
        self.predicate_inst.generate_new_preds()
        transfer = []
        modified_src_tree = []
        for i in range(0, len(individual_trees)):
            mod_src_tree, transfer = self.mapping_tree(individual_trees[i],
                                                       source_trees[i], 
                                                       i, transfer, ind)
            modified_src_tree.append(mod_src_tree)
        transfer = self.include_parameters(transfer)
        return modified_src_tree, transfer
