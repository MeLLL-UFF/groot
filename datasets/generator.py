'''
   Functions to return datasets in file folder
   Name:         generator.py
   Author:       Leticia Freire
   Updated:      October 1st, 2020
   License:      GPLv3
   Inspired by Rodrigo Azevedo's TreeBostler code
   You can see here: https://github.com/MeLL-UFF/TreeBoostler
'''

import os
import sys
import time

from get_datasets import *
import numpy as np
import random
import json

def open_files():
  experiments = []
  for line in open('experiments.json', 'r'):  
    experiments.append(json.loads(line))         


  with open('files/json/bk_experiments.json', 'r') as outfile: 
    bk_experiments  = json.load(outfile)           

  return experiments, bk_experiments 


def generate_file(bk, source, predicate, results):
  # Load source dataset
  src_total_data = datasets.load(source, bk[source], seed=results['save']['seed'])
  src_data = datasets.load(source, bk[source], target=predicate, balanced=results['source_balanced'], seed=results['save']['seed'])  

  # Group and shuffle
  src_facts = datasets.group_folds(src_data[0])
  src_pos = datasets.group_folds(src_data[1])
  src_neg = datasets.group_folds(src_data[2])

  print('Source train facts examples: %s' % len(src_facts))
  print('Source train pos examples: %s' % len(src_pos))
  print('Source train neg examples: %s\n' % len(src_neg))

  dir_path = 'files/' + source + '/src/'
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  with open(dir_path +'/src_facts.txt', 'w') as file:
    for fact in src_facts:
      file.write(fact)
      file.write('\n')
  file.close()

  with open(dir_path + '/src_pos.txt', 'w') as file:
    for pos in src_pos:
      file.write(pos)
      file.write('\n')
  file.close()

  with open(dir_path + '/src_neg.txt', 'w') as file:
    for neg in src_neg:
      file.write(neg)
      file.write('\n')
  file.close()


def generate_train_test_files():
  experiments, bk_experiments = open_files()

  for experiment in experiments:
    print(experiment)
    results = { 'save': { }}
    firstRun = True
    if firstRun:
        results['save'] = {
            'experiment': 0,
            'n_runs': 0,
            'seed': 441773,
            'source_balanced' : 1,
            'balanced' : 1,
            'folds' : 3,      
            'nodeSize' : 2,
            'numOfClauses' : 8,
            'maxTreeDepth' : 3
            }


    source_balanced = 1
    balanced = 1


    source = experiment['source']
    target = experiment['target']
    predicate = experiment['predicate']
    to_predicate = experiment['to_predicate']

    # Load source dataset
    generate_file(bk_experiments, source, predicate, results)
    generate_file(bk_experiments, target, to_predicate, results)
