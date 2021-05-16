# -*- coding: utf-8 -*-
#
# eval.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os
import logging
import time
import pickle
import dgl
import torch
import numpy as np
import random
import json
from utils import get_compatible_batch_size

from dataloader import EvalDatasetNew, TrainDataset
from dataloader import get_dataset
from wiki2vec.dictionary import Dictionary

backend = os.environ.get('DGLBACKEND', 'pytorch')
if backend.lower() == 'mxnet':
    import multiprocessing as mp
    from train_mxnet import load_model_from_checkpoint
    from train_mxnet import test
else:
    import torch.multiprocessing as mp
    from train_pytorch import load_model_from_checkpoint
    from train_pytorch import test, test_mp, test_rel_constrain_demo

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--model_name', default='TransE',
                          choices=['TransE', 'TransE_l1', 'TransE_l2', 'TransR',
                                   'RESCAL', 'DistMult', 'ComplEx', 'RotatE'],
                          help='The models provided by DGL-KE.')
        self.add_argument('--data_path', type=str, default='data',
                          help='The path of the directory where DGL-KE loads knowledge graph data.')
        self.add_argument('--dataset', type=str, default='FB15k',
                          help='The name of the builtin knowledge graph. Currently, the builtin knowledge '\
                                  'graphs include FB15k, FB15k-237, wn18, wn18rr and Freebase. '\
                                  'DGL-KE automatically downloads the knowledge graph and keep it under data_path.')
        self.add_argument('--format', type=str, default='built_in',
                          help='The format of the dataset. For builtin knowledge graphs,'\
                                  'the foramt should be built_in. For users own knowledge graphs,'\
                                  'it needs to be raw_udd_{htr} or udd_{htr}.')
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='A list of data file names. This is used if users want to train KGE'\
                                  'on their own datasets. If the format is raw_udd_{htr},'\
                                  'users need to provide train_file [valid_file] [test_file].'\
                                  'If the format is udd_{htr}, users need to provide'\
                                  'entity_file relation_file train_file [valid_file] [test_file].'\
                                  'In both cases, valid_file and test_file are optional.')
        self.add_argument('--test-triples-file', type=str, required=True, help='file containing triples to evaluate on')
        self.add_argument('--model_path', type=str, default='ckpts',
                          help='The path of the directory where models are saved.')
        self.add_argument('--batch_size_eval', type=int, default=8,
                          help='The batch size used for evaluation.')
        self.add_argument('--neg_sample_size_eval', type=int, default=-1,
                          help='The negative sampling size for evaluation.')
        self.add_argument('--neg_deg_sample_eval', action='store_true',
                          help='Negative sampling proportional to vertex degree for evaluation.', default=False)
        self.add_argument('--hidden_dim', type=int, default=256,
                          help='The hidden dim used by relation and entity')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='The margin value in the score function. It is used by TransX and RotatE.')
        self.add_argument('--eval_percent', type=float, default=1,
                          help='The percentage of data used for evaluation.')
        self.add_argument('--no_eval_filter', action='store_true',
                          help='Disable filter positive edges from randomly constructed negative edges for evaluation')
        self.add_argument('--gpu', type=int, default=[-1], nargs='+',
                          help='a list of active gpu ids, e.g. 0')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='Evaluate a knowledge graph embedding model with both CPUs and GPUs.'\
                                  'The embeddings are stored in CPU memory and the training is performed in GPUs.'\
                                  'This is usually used for training a large knowledge graph embeddings.')
        self.add_argument('-de', '--double_ent', action='store_true',
                          help='Double entitiy dim for complex number It is used by RotatE.')
        self.add_argument('-dr', '--double_rel', action='store_true',
                          help='Double relation dim for complex number.')
        self.add_argument('--num_proc', type=int, default=1,
                          help='The number of processes to evaluate the model in parallel.'\
                                  'For multi-GPU, the number of processes by default is set to match the number of GPUs.'\
                                  'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')
        self.add_argument('--num_thread', type=int, default=1,
                          help='The number of CPU threads to evaluate the model in each process.'\
                                  'This argument is used for multiprocessing computation.')
        self.add_argument("--seed", type=int, default=11117)
        self.add_argument("--wiki-name-file", type=str, default=None)
        self.add_argument('--rel-type-dict-file', type=str, required=True)
        self.add_argument('--entity-child-dict-file', type=str, required=True)
        self.add_argument('--sampler-type', type=str, choices=['head', 'tail', 'both'], required=True)
        self.add_argument('--dictionary-file', type=str, required=True, help='name of dictionary file')
        self.add_argument('--wiki-link-file', type=str, required=True, help='path to wikipedia links file')

    def parse_args(self):
        args = super().parse_args()
        return args

def get_logger(args):
    if not os.path.exists(args.model_path):
        raise Exception('No existing model_path: ' + args.model_path)

    log_file = os.path.join(args.model_path, 'eval.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    logger = logging.getLogger(__name__)
    print("Logs are being recorded at: {}".format(log_file))
    return logger

def main():
    args = ArgParser().parse_args()
    print('args.neg_deg_sample_eval = {}'.format(args.neg_deg_sample_eval))
    print('args.neg_sample_size_eval = {}'.format(args.neg_sample_size_eval))
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dgl.random.seed(args.seed)

    ent_child_dict = json.load(open(args.entity_child_dict_file, 'r'))
    rel_type_dict = pickle.load(open(args.rel_type_dict_file, 'rb'))

    wiki_name = None
    if args.wiki_name_file:
        wiki_name = json.load(open(args.wiki_name_file))

    args.eval_filter = not args.no_eval_filter
    if args.neg_deg_sample_eval:
        assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

    wiki_link_dict = json.load(open(args.wiki_link_file)) # mapping from wikidata entity IDs to wikipedia entity names
    dictionary = Dictionary.load(args.dictionary_file)

    # load dataset and samplers
    dataset = get_dataset(args.data_path, args.dataset, args.format, dictionary, wiki_link_dict, args.data_files)
    args.train = False
    args.valid = False
    args.test = True
    args.strict_rel_part = False
    args.soft_rel_part = False
    args.async_update = False
    if len(args.gpu) > 1:
        args.mix_cpu_gpu = True
        if args.num_proc < len(args.gpu):
            args.num_proc = len(args.gpu)
    # We need to ensure that the number of processes should match the number of GPUs.
    if len(args.gpu) > 1 and args.num_proc > 1:
        assert args.num_proc % len(args.gpu) == 0, \
                'The number of processes needs to be divisible by the number of GPUs'

    logger = get_logger(args)
    # Here we want to use the regualr negative sampler because we need to ensure that
    # all positive edges are excluded.
    eval_dataset = EvalDatasetNew(dataset, args, rel_constrain=True)

    if args.neg_sample_size_eval < 0:
        args.neg_sample_size_eval = args.neg_sample_size = eval_dataset.g.number_of_nodes()
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)

    id2entity_map = {val:key for key,val in dataset.entity2id.items()}
    id2relation_map = {val:key for key,val in dataset.relation2id.items()}

    # print('args.neg_sample_size_eval = {}'.format(args.neg_sample_size_eval))
    print(eval_dataset.g.number_of_nodes())
    print(args.neg_sample_size_eval)

    args.num_workers = 8 # fix num_workers to 8
    if args.num_proc > 1:
        test_sampler_tails = []
        test_sampler_heads = []
        for i in range(args.num_proc):
            test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.eval_filter,
                                                            mode='chunk-head',
                                                            num_workers=args.num_workers,
                                                            rank=i, ranks=args.num_proc)
            test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.eval_filter,
                                                            mode='chunk-tail',
                                                            num_workers=args.num_workers,
                                                            rank=i, ranks=args.num_proc)
            test_sampler_heads.append(test_sampler_head)
            test_sampler_tails.append(test_sampler_tail)
    else:
        if args.sampler_type in ['head', 'both']:
            test_sampler_head = eval_dataset.create_rel_constrain_sampler('test', args.batch_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.eval_filter,
                                                            ent_child_dict,
                                                            rel_type_dict,
                                                            mode='chunk-head',
                                                            num_workers=args.num_workers,
                                                            rank=0, ranks=1)
        if args.sampler_type in ['tail', 'both']:
            test_sampler_tail = eval_dataset.create_rel_constrain_sampler('test', args.batch_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.eval_filter,
                                                            ent_child_dict,
                                                            rel_type_dict,
                                                            mode='chunk-tail',
                                                            num_workers=args.num_workers,
                                                            rank=0, ranks=1)

    # load model
    n_entities = dataset.n_entities
    n_relations = dataset.n_relations
    ckpt_path = args.model_path
    model = load_model_from_checkpoint(logger, args, n_entities, n_relations, ckpt_path)

    if args.num_proc > 1:
        model.share_memory()
    # test
    args.step = 0
    args.max_step = 0
    start = time.time()
    if args.num_proc > 1:
        queue = mp.Queue(args.num_proc)
        procs = []
        for i in range(args.num_proc):
            proc = mp.Process(target=test_mp, args=(args,
                                                    model,
                                                    [test_sampler_heads[i], test_sampler_tails[i]],
                                                    i,
                                                    'Test',
                                                    queue))
            procs.append(proc)
            proc.start()

        total_metrics = {}
        metrics = {}
        logs = []
        for i in range(args.num_proc):
            log = queue.get()
            logs = logs + log

        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        print("-------------- Test result --------------")
        for k, v in metrics.items():
            print('Test average {}: {}'.format(k, v))
        print("-----------------------------------------")

        for proc in procs:
            proc.join()
    else:
        if args.sampler_type == 'both':
            test_rel_constrain_demo(args, model, [test_sampler_head, test_sampler_tail], id2entity_map, id2relation_map, wiki_name)
        elif args.sampler_type == 'head':
            test_rel_constrain_demo(args, model, [test_sampler_head], id2entity_map, id2relation_map, wiki_name)
        elif args.sampler_type == 'tail':
            test_rel_constrain_demo(args, model, [test_sampler_tail], id2entity_map, id2relation_map, wiki_name)

    print('Test takes {:.3f} seconds'.format(time.time() - start))

if __name__ == '__main__':
    main()
