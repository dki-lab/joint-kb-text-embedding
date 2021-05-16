# -*- coding: utf-8 -*-
#
# train.py
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
import math
import os
import logging
import logging.handlers
import time
import numpy as np
from multiprocessing import set_start_method
from tqdm import tqdm
import subprocess
import joblib
import dgl

from dataloader import EvalDataset, TrainDataset, NewBidirectionalOneShotIterator
from dataloader import get_dataset

from utils import get_compatible_batch_size, save_model, CommonArgParser

from wiki2vec.dictionary import Dictionary

backend = os.environ.get('DGLBACKEND', 'pytorch')
import torch.multiprocessing as mp
from train_pytorch import load_model
from train_pytorch import train_ke, train_mp_ke
from train_pytorch import test, test_mp

# import dist_train
from skipGramModel import *
from torch import optim
import json
import multiprocessing
import random

class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--gpu', type=int, default=[-1], nargs='+', 
                          help='A list of gpu ids, e.g. 0 1 2 4')
        self.add_argument("--cuda", action='store_true', default=False, help="enable cuda")
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='Training a knowledge graph embedding model with both CPUs and GPUs.'\
                                  'The embeddings are stored in CPU memory and the training is performed in GPUs.'\
                                  'This is usually used for training a large knowledge graph embeddings.')
        self.add_argument('--valid', action='store_true',
                          help='Evaluate the model on the validation set in the training.')
        self.add_argument('--rel_part', action='store_true',
                          help='Enable relation partitioning for multi-GPU training.')
        self.add_argument('--async_update', action='store_true',
                          help='Allow asynchronous update on node embedding for multi-GPU training.'\
                                  'This overlaps CPU and GPU computation to speed up.')
        self.add_argument('--dump-db-file', type=str, required=True, help='name of output db file')
        self.add_argument('--dictionary-file', type=str, required=True, help='name of dictionary file')
        self.add_argument('--mention-db-file', type=str, required=True, help='name of mention db file')
        self.add_argument('--link-graph-file', type=str, default=None, help='name of link graph file')
        self.add_argument('--entities-per-page', type=int, default=10, help='For processing each page, the '
                  'specified number of randomly chosen entities are used to predict their '
                  'neighboring entities in the link graph')
        self.add_argument('--window', type=int, default=5, help='size of window for skip-gram')
        self.add_argument('--negative', type=int, default=15, help='no. of negatives for skip-gram')
        self.add_argument('--sg_batch_size', type=int, default=100, help='batch size for skip gram model')
        self.add_argument('--sg_lr', type=float, default=0.025, help='learning rate for skip gram model') #TODO: check LR initial value
        self.add_argument("--sample", type=float, default=1e-4, help="subsample threshold")
        self.add_argument('--num_proc_train', type=int, default=-1, help='no. of training CPU/GPU proc.')
        # self.add_argument("--sg_iters", type=int, default=5, help="no. of iters of SG model")
        self.add_argument("--n_iters", type=int, default=5, help="no. of iters of combined model")
        self.add_argument("--timeout", type=int, default=1000, help="time (in sec.) to wait before for incoming data batches before exiting training")
        self.add_argument("--reg_coeff", type=float, default=100.0, help="value of reg. coeff.")
        self.add_argument("--seed", type=int, default=11117)
        self.add_argument("--transe-entity-ckpt-path", type=str, default=None)
        self.add_argument("--transe-relation-ckpt-path", type=str, default=None)
        self.add_argument("--sg-ckpt-emb0-path", type=str, default=None)
        self.add_argument("--sg-ckpt-emb1-path", type=str, default=None)
        self.add_argument("--reg-loss-start-epoch", type=int, default=1)
        self.add_argument("--use-input-embedding", action='store_true', default=False)
        self.add_argument("--start-epoch", type=int, default=0)
        self.add_argument("--proj-layer-ckpt-path", type=str, default=None)
        self.add_argument("--sg-reg-optimizer", type=str, choices=['sgd', 'adam'])
        self.add_argument("--kb-process-method", type=str, choices=['forkserver', 'fork'], default='fork')
        self.add_argument('--wiki-link-file', type=str, required=True, help='path to wikipedia links file')
        

def listener(q, args):
    '''listens for messages on the q, writes to file. '''
    if args.start_epoch==0:
      loss_log_file = os.path.join(args.save_path, 'train_loss.log')
    else:
      loss_log_file = os.path.join(args.save_path, 'train_loss_{}.log'.format(args.start_epoch))

    with open(loss_log_file, 'w') as f:
        log_id = 0
        while 1:
            m = q.get()
            if m == 'kill':
                f.write('end of log')
                f.flush()
                break
            f.write(str(m) + '\n')
            if log_id % 1000 == 0:
              f.flush()
            log_id += 1


def main():
    args = ArgParser().parse_args()

    print(args.use_input_embedding)
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dgl.random.seed(args.seed)

    logging.basicConfig(format='%(name)s - %(message)s', level=logging.INFO)
    logger = multiprocessing.get_logger()
    logger.warning('This will get logged to a file')

    # queue for log messages
    manager = mp.Manager()
    log_queue = manager.Queue()

    listener_process = multiprocessing.Process(target=listener,
                                       args=(log_queue, args))
    listener_process.start()

    set_start_method('forkserver', force=True) # added for SG model

    # restrict no. of processes for TransE model to 8
    vars(args)['num_proc_sg'] = args.num_proc
    if args.num_proc >=8:
      args.num_proc = 8

    if args.num_proc_train == -1:
        args.num_proc_train = args.num_proc_sg
        
    init_time_start = time.time()
    # load dataset and samplers
    dataset = get_dataset(args.data_path, args.dataset, args.format, args.data_files)
    dictionary = Dictionary.load(args.dictionary_file)
    
    if args.neg_sample_size_eval < 0:
        args.neg_sample_size_eval = dataset.n_entities
    args.batch_size = get_compatible_batch_size(args.batch_size, args.neg_sample_size)
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)
    # We should turn on mix CPU-GPU training for multi-GPU training.
    if len(args.gpu) > 1:
        args.mix_cpu_gpu = True
        if args.num_proc < len(args.gpu):
            args.num_proc = len(args.gpu)
    # We need to ensure that the number of processes should match the number of GPUs.
    if len(args.gpu) > 1 and args.num_proc > 1:
        assert args.num_proc % len(args.gpu) == 0, \
                'The number of processes needs to be divisible by the number of GPUs'
    # For multiprocessing training, we need to ensure that training processes are synchronized periodically.
    if args.num_proc > 1:
        args.force_sync_interval = 1000

    args.eval_filter = not args.no_eval_filter
    if args.neg_deg_sample_eval:
        assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

    args.soft_rel_part = args.mix_cpu_gpu and args.rel_part
    train_data = TrainDataset(dataset, args, ranks=args.num_proc)

    print('no. of nodes = {}'.format(train_data.g.number_of_nodes()))
    print('no. of edges = {}'.format(train_data.g.number_of_edges()))

    if args.num_proc > 1:
        train_samplers = []
        for i in range(args.num_proc):
            # print('rank = {}'.format(i))
            train_sampler_head = train_data.create_sampler(args.batch_size,
                                                           args.neg_sample_size,
                                                           args.neg_sample_size,
                                                           mode='head',
                                                           num_workers=args.num_proc,
                                                           shuffle=True,
                                                           exclude_positive=False,
                                                           rank=i)
            train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                           args.neg_sample_size,
                                                           args.neg_sample_size,
                                                           mode='tail',
                                                           num_workers=args.num_proc,
                                                           shuffle=True,
                                                           exclude_positive=False,
                                                           rank=i)
            train_samplers.append(NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                                  args.neg_sample_size, args.neg_sample_size,
                                                                  True, dataset.n_entities))

    else: # This is used for debug
        train_sampler_head = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_sample_size,
                                                       mode='head',
                                                       num_workers=args.num_proc,
                                                       shuffle=True,
                                                       exclude_positive=False)
        train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_sample_size,
                                                       mode='tail',
                                                       num_workers=args.num_proc,
                                                       shuffle=True,
                                                       exclude_positive=False)
        train_sampler = NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                        args.neg_sample_size, args.neg_sample_size,
                                                        True, dataset.n_entities)

    # if there is no cross partition relaiton, we fall back to strict_rel_part
    args.strict_rel_part = args.mix_cpu_gpu and (train_data.cross_part == False)
    args.num_workers = 8 # fix num_worker to 8

    if args.valid or args.test:
        if len(args.gpu) > 1:
            args.num_test_proc = args.num_proc if args.num_proc < len(args.gpu) else len(args.gpu)
        else:
            args.num_test_proc = args.num_proc
        if args.valid:
            assert dataset.valid is not None, 'validation set is not provided'
        if args.test:
            assert dataset.test is not None, 'test set is not provided'
        eval_dataset = EvalDataset(dataset, args)

    if args.valid:
        if args.num_proc > 1:
            valid_sampler_heads = []
            valid_sampler_tails = []
            for i in range(args.num_proc):
                # print('rank = {}'.format(i))
                valid_sampler_head = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                                  args.neg_sample_size_eval,
                                                                  args.neg_sample_size_eval,
                                                                  args.eval_filter,
                                                                  mode='chunk-head',
                                                                  num_workers=args.num_proc,
                                                                  rank=i, ranks=args.num_proc)
                valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                                  args.neg_sample_size_eval,
                                                                  args.neg_sample_size_eval,
                                                                  args.eval_filter,
                                                                  mode='chunk-tail',
                                                                  num_workers=args.num_proc,
                                                                  rank=i, ranks=args.num_proc)
                valid_sampler_heads.append(valid_sampler_head)
                valid_sampler_tails.append(valid_sampler_tail)
        else: # This is used for debug
            valid_sampler_head = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                             args.neg_sample_size_eval,
                                                             args.neg_sample_size_eval,
                                                             args.eval_filter,
                                                             mode='chunk-head',
                                                             num_workers=args.num_proc,
                                                             rank=0, ranks=1)
            valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                             args.neg_sample_size_eval,
                                                             args.neg_sample_size_eval,
                                                             args.eval_filter,
                                                             mode='chunk-tail',
                                                             num_workers=args.num_proc,
                                                             rank=0, ranks=1)

    if args.test:
        if args.num_test_proc > 1:
            test_sampler_tails = []
            test_sampler_heads = []
            for i in range(args.num_test_proc):
                test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.eval_filter,
                                                                 mode='chunk-head',
                                                                 num_workers=1,
                                                                 rank=i, ranks=args.num_test_proc)
                test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.eval_filter,
                                                                 mode='chunk-tail',
                                                                 num_workers=1,
                                                                 rank=i, ranks=args.num_test_proc)
                test_sampler_heads.append(test_sampler_head)
                test_sampler_tails.append(test_sampler_tail)
        else:
            test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.eval_filter,
                                                            mode='chunk-head',
                                                            num_workers=1,
                                                            rank=0, ranks=1)
            test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.eval_filter,
                                                            mode='chunk-tail',
                                                            num_workers=1,
                                                            rank=0, ranks=1)

    # load model
    model = load_model(logger, args, dataset.n_entities, dataset.n_relations) # KE model

    # load transE ckpt
    if args.transe_entity_ckpt_path:
      model.entity_emb.emb = torch.Tensor(np.load(args.transe_entity_ckpt_path))
    
    if args.transe_relation_ckpt_path:
      model.relation_emb.emb = torch.Tensor(np.load(args.transe_relation_ckpt_path))

    sg_model = skipGramModel(dictionary.word_size+dictionary.entity_size, args.hidden_dim, args.window, args.negative)
    
    if args.cuda:
        sg_model = sg_model.cuda()
    
    sg_model.share_memory()

    # load skip-gram ckpt
    if args.sg_ckpt_emb0_path:
      sg_model.emb0_lookup.weight.data.copy_(torch.Tensor(np.load(args.sg_ckpt_emb0_path)))

    if args.sg_ckpt_emb1_path:
      sg_model.emb1_lookup.weight.data.copy_(torch.Tensor(np.load(args.sg_ckpt_emb1_path)))

    if args.proj_layer_ckpt_path:
      model.proj_layer.load_state_dict(torch.load(args.proj_layer_ckpt_path))

    sg_reg_optimizer_list = []
    for i in range(args.num_proc):
      if args.sg_reg_optimizer == 'sgd':
        sg_reg_optimizer_list.append(optim.SGD(sg_model.parameters(), lr=args.sg_lr * (1 - word_count_actual.value / (args.n_iters * word_count))))
      else:
        sg_reg_optimizer_list.append(optim.SparseAdam(sg_model.parameters()))

    # proj_layer = nn.Linear(model.entity_dim, model.entity_dim)
    proj_layer_optimizer_list = []

    for i in range(args.num_proc):
      proj_layer_optimizer_list.append(optim.Adam(model.proj_layer.parameters()))

    if args.num_proc > 1 or args.async_update:
        model.share_memory()
        model.proj_layer.share_memory()

    # print('entity2id = {}'.format(dataset.entity2id))
    # print('relation2id = {}'.format(dataset.relation2id))
    id2entity_map = {val:key for key,val in dataset.entity2id.items()}

    # We need to free all memory referenced by dataset.
    # eval_dataset = None
    # dataset = None
    
    # table_ptr_val = data_producer.init_unigram_table(word_list, freq, word_count) # initialize unigram table
    wiki_link_dict = json.load(open(args.wiki_link_file)) # mapping from wikidata entity IDs to wikipedia entity names

    print('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))
    
    # train
    start = time.time()
    rel_parts = train_data.rel_parts if args.strict_rel_part or args.soft_rel_part else None
    cross_rels = train_data.cross_rels if args.soft_rel_part else None
    max_step_per_epoch = int(train_data.g.number_of_edges()/ (args.batch_size * args.num_proc))+1
    vars(args)['max_step_per_epoch'] = max_step_per_epoch
    print('max_step_per_epoch = {}'.format(max_step_per_epoch))

    # TODO: change iter_id
    ctx = torch.multiprocessing.get_context(args.kb_process_method)
    # ctx = torch.multiprocessing.get_context('forkserver')
    # ctx = torch.multiprocessing.get_context('spawn')
    os.environ['TMPDIR'] = args.save_path
    
    common_node_ids_dict_idx = list(filter(lambda x:x[1]!=-1, map(lambda x:(x[0], dictionary.get_entity_index(x[1])), filter(lambda x: x[1], map(lambda x:(x[0], wiki_link_dict.get(x[1], None)), list(id2entity_map.items()))))))
    # print(list(common_node_ids_dict_idx))
    common_node_ids = [x[0] for x in common_node_ids_dict_idx]
    common_node_ids_tensor = torch.tensor(common_node_ids)

    common_entity_sg_idx = torch.tensor([x[1] for x in common_node_ids_dict_idx])

    print('common_node_ids_tensor = {}'.format(common_node_ids_tensor))
    print('common_entity_sg_idx = {}'.format(common_entity_sg_idx))

    for iter_id in range(args.start_epoch, args.n_iters):
        epoch_start_time = time.time()

        sg_model.cpu()
        
        log_queue.put('Epoch {} of TransE model'.format(iter_id))
        #***********************************
        # train 1 epoch of TransE model
        #***********************************
        
        if args.num_proc > 1:
            procs = []
            barrier = ctx.Barrier(args.num_proc)
            for i in range(args.num_proc):
                valid_sampler = [valid_sampler_heads[i], valid_sampler_tails[i]] if args.valid else None
                # proc = mp.Process(target=train_process, args=(p_id, dump_db, dictionary, tokenizer, word_list, word_count_actual, freq, args, model))
                proc = ctx.Process(target=train_mp_ke, args=(args, iter_id, args.reg_loss_start_epoch, args.use_input_embedding, model, sg_model, log_queue, sg_reg_optimizer_list[i], proj_layer_optimizer_list[i], id2entity_map, wiki_link_dict, train_samplers[i], valid_sampler, i, rel_parts, cross_rels, barrier))
                procs.append(proc)
                proc.start()
                print('TransE proc {} started'.format(i))
            for i,proc in enumerate(procs):
                proc.join()
                print('TransE proc {} joined'.format(i))
                # print(model.proj_layer.bias)

        else:
          valid_samplers = [valid_sampler_head, valid_sampler_tail]
          train_ke(args, iter_id, args.reg_loss_start_epoch, args.use_input_embedding, model, sg_model, log_queue, sg_reg_optimizer_list[0], proj_layer_optimizer_list[0], id2entity_map, wiki_link_dict, train_sampler, valid_samplers, 0, rel_parts, cross_rels)
        
        print('Iteration {} of TransE model completed in {} sec.'.format(iter_id, time.time()-epoch_start_time))
        
        model.entity_emb.save(args.save_path, args.dataset+'_'+model.model_name+'_entity')
        model.relation_emb.save(args.save_path, args.dataset+'_'+model.model_name+'_relation')
        # print(model.proj_layer.bias)
        # print(model.proj_layer.state_dict()['bias'])
        
        torch.save(model.proj_layer.state_dict(), os.path.join(args.save_path, 'proj_layer'+'.pt'))
        
        # TODO: copy embeddings from transE to skip-gram
        sg_model.emb0_lookup.weight.data[common_entity_sg_idx] = model.entity_emb.emb[common_node_ids_tensor]

        # save skip-gram pytorch model weights to disk for use by wikipedia2vec API

        # sg_emb0_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb0_sg'+'_{}'.format(iter_id)+'.npy') # save intermediate emb0 embedding
        sg_emb0_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb0_sg'+'.npy')
        np.save(sg_emb0_fname, sg_model.emb0_lookup.weight.data.cpu().numpy())

        # sg_emb1_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb1_sg'+'_{}'.format(iter_id)+'.npy') # save intermediate emb1 embedding
        sg_emb1_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb1_sg'+'.npy')
        np.save(sg_emb1_fname, sg_model.emb1_lookup.weight.data.cpu().numpy())

        epoch_start_time = time.time()

        log_queue.put('Epoch {} of skip-gram model'.format(iter_id))
        #***********************************
        # train 1 epoch of SG model
        #***********************************

        subprocess.run(['wikinew', 'train-embedding', args.dump_db_file, args.dictionary_file, sg_emb0_fname, sg_emb1_fname, str(args.n_iters), str(iter_id), os.path.join(args.save_path, 'emb_file'), '--pool-size', str(args.num_proc_train), '--dim-size', str(args.hidden_dim), '--mention-db', args.mention_db_file, '--link-graph', args.link_graph_file])
        print('Iteration {} of skip-gram model completed in {} sec.'.format(iter_id, time.time()-epoch_start_time))
        # torch.cuda.synchronize()

        # copy skip-gram model weights from 'model_file' to PyTorch model
        emb_combined = joblib.load(os.path.join(args.save_path, 'emb_file'))
        sg_model.emb0_lookup.weight.data.copy_(torch.tensor(emb_combined['syn0']))
        sg_model.emb1_lookup.weight.data.copy_(torch.tensor(emb_combined['syn1']))

        # sg_emb0_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb0_sg'+'_{}'.format(iter_id)+'.npy')
        sg_emb0_iter_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb0_sg'+'_{}'.format(iter_id)+'.npy')
        sg_emb0_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb0_sg'+'.npy')
        np.save(sg_emb0_fname, sg_model.emb0_lookup.weight.data.cpu().numpy())

        # sg_emb1_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb1_sg'+'_{}'.format(iter_id)+'.npy')
        sg_emb1_iter_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb1_sg'+'_{}'.format(iter_id)+'.npy')
        sg_emb1_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb1_sg'+'.npy')
        np.save(sg_emb1_fname, sg_model.emb1_lookup.weight.data.cpu().numpy())

        # TODO: copy embeddings from skip-gram to TransE
        model.entity_emb.emb[common_node_ids_tensor] = sg_model.emb0_lookup.weight.data[common_entity_sg_idx]

    print('training takes {} seconds'.format(time.time() - start))
    model.entity_emb.save(args.save_path, args.dataset+'_'+model.model_name+'_entity')
    model.relation_emb.save(args.save_path, args.dataset+'_'+model.model_name+'_relation')
    
    sg_emb0_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb0_sg'+'.npy')
    np.save(sg_emb0_fname, sg_model.emb0_lookup.weight.data.cpu().numpy())

    sg_emb1_fname = os.path.join(args.save_path, args.dataset+'_'+model.model_name+'_emb1_sg'+'.npy')
    np.save(sg_emb1_fname, sg_model.emb1_lookup.weight.data.cpu().numpy())
        
    log_queue.put("kill")
    listener_process.join()
    # if not args.no_save_emb:
        # save_model(args, model)
    
if __name__ == '__main__':
    main()
