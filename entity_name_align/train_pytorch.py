# -*- coding: utf-8 -*-
#
# train_pytorch.py
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
import itertools
import requests
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.optim as optim
import torch as th
import numpy as np

from distutils.version import LooseVersion
TH_VERSION = LooseVersion(th.__version__)
if TH_VERSION.version[0] == 1 and TH_VERSION.version[1] < 2:
    raise Exception("DGL-ke has to work with Pytorch version >= 1.2")
from models.pytorch.tensor_models import thread_wrapped_func
from models import KEModel
from utils import save_model, get_compatible_batch_size

import sys, os
import logging
import time
from functools import wraps

import dgl
import dgl.backend as F

from dataloader import EvalDataset, NewBidirectionalOneShotIterator
from dataloader import get_dataset
from tqdm import tqdm
import queue
import multiprocessing
import torch
import random

# cython module import
from wiki2vec.dictionary import Dictionary

def load_model(logger, args, n_entities, n_relations, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel)
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"
    return model

def load_model_from_checkpoint(logger, args, n_entities, n_relations, ckpt_path):
    model = load_model(logger, args, n_entities, n_relations)
    model.load_emb(ckpt_path, args.dataset)
    return model


def train_ke(args, iter_id, use_name_graph, reg_loss_start_epoch, use_input_embedding, model, sg_model, log_queue, sg_optimizer, proj_layer_optimizer, id2entity_map, wiki_link_dict, train_sampler, valid_samplers, rank=0, rel_parts=None, cross_rels=None, barrier=None, client=None):
    print('iter_id = {}'.format(iter_id))
    logs = []
    # for arg in vars(args):
        # logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.async_update:
        model.create_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.prepare_relation(th.device('cuda:' + str(gpu_id)))
    if args.soft_rel_part:
        model.prepare_cross_rels(cross_rels)

    print('rank = {}'.format(rank))

    train_start = start = time.time()
    sample_time = 0
    update_time = 0
    forward_time = 0
    backward_time = 0
    
    dictionary = Dictionary.load(args.dictionary_file)

    #******************************************************************************
    # train KE model
    #******************************************************************************
    
    # print('start of epoch')
    # print(model.proj_layer.bias)

    for step in range(0, args.max_step_per_epoch):
        # print('step = {}'.format(step))
        start1 = time.time()
        # print(train_sampler)
        pos_g, neg_g = next(train_sampler)

        # print('pos_g = {}'.format(pos_g))
        # print('no. of nodes = {}'.format(pos_g.number_of_nodes()))
        # print('no. of edges = {}'.format(pos_g.number_of_edges()))

        # print('is_multigraph = {}'.format(pos_g.is_multigraph))

        sample_time += time.time() - start1

        if client is not None:
            model.pull_model(client, pos_g, neg_g)

        start1 = time.time()
        
        proj_layer_optimizer.zero_grad()

        loss, log = model.forward(pos_g, neg_g, gpu_id) # KE model update
        # loss = 0
        # log = {}
        # print('pos_g.ndata[emb] = {}'.format(pos_g.ndata['emb'].size()))
        # print('pos_g.ndata[id] = {}'.format(pos_g.ndata['id'].size()))

        # title_list = list(filter(lambda x: x, map(lambda x:wiki_link_dict.get(id2entity_map[x.item()], None), pos_g.ndata['id'])))
        
        forward_time += time.time() - start1
        sg_optimizer.zero_grad()

        start1 = time.time()
        # print('ke loss = {}'.format(loss))

        # loss_combined = loss + args.reg_coeff*reg_loss

        log_queue.put("Loss/KE/{}: step id:{} = {}".format(str(rank), step, loss))

        if use_name_graph:
            loss_combined = args.reg_coeff * loss
        else:
            loss_combined = loss # in first epoch, no alignment takes place

        loss_combined.backward()

        backward_time += time.time() - start1

        start1 = time.time()
        if client is not None:
            model.push_gradient(client)
        else:
            model.update(gpu_id)

        
        # print('training KE model')
        update_time += time.time() - start1
        logs.append(log)
        # if step>10:
            # break
        # force synchronize embedding across processes every X steps
        
        if args.force_sync_interval > 0 and (step + 1) % args.force_sync_interval == 0:
            barrier.wait()

        if (step + 1) % args.log_interval == 0:
            if (client is not None) and (client.get_machine_id() != 0):
                pass
            else:
                for k in logs[0].keys():
                    v = sum(l[k] for l in logs) / len(logs)
                    print('[proc {}][Train]({}/{}) average {}: {}'.format(rank, (step + 1), args.max_step, k, v))
                logs = []
                print('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, args.log_interval,
                                                                time.time() - start))
                print('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                    rank, sample_time, forward_time, backward_time, update_time))
                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0
                start = time.time()

        # update sg optimizer LR
        lr = args.sg_lr * (1 - (iter_id * args.max_step_per_epoch + step+1) / (args.n_iters * args.max_step_per_epoch))
        if lr < 0.0001 * args.sg_lr:
            lr = 0.0001 * args.sg_lr
        for param_group in sg_optimizer.param_groups:
            param_group['lr'] = lr
        

    # print('end of epoch')
    # print(model.proj_layer.bias)
    # print(model.proj_layer.bias)

    
    if args.valid and valid_samplers is not None:
        valid_start = time.time()
        if args.strict_rel_part or args.soft_rel_part:
            model.writeback_relation(rank, rel_parts)
        # forced sync for validation
        if barrier is not None:
            barrier.wait()
        test(args, model, valid_samplers, rank, mode='Valid')
        print('[proc {}]validation take {:.3f} seconds:'.format(rank, time.time() - valid_start))
        if args.soft_rel_part:
            model.prepare_cross_rels(cross_rels)
        if barrier is not None:
            barrier.wait()
            # break
        
    if args.async_update:
        model.finish_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.writeback_relation(rank, rel_parts)
    
    # print('complete end of epoch')
    # print(model.proj_layer.bias)


def test(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

    with th.no_grad():
        logs = []
        for sampler in test_samplers:
            for pos_g, neg_g in sampler:
                model.forward_test(pos_g, neg_g, logs, gpu_id)

        metrics = {}
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        if queue is not None:
            queue.put(logs)
        else:
            for k, v in metrics.items():
                print('[{}]{} average {}: {}'.format(rank, mode, k, v))
    test_samplers[0] = test_samplers[0].reset()
    test_samplers[1] = test_samplers[1].reset()

def test_rel_constrain(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

    with th.no_grad():
        logs = []
        for sampler_dict in test_samplers: # head or tail sampler dict
            log_len = 0
            for r in sampler_dict:
                print('r = {}'.format(r))
                print(type(sampler_dict[r]))
                print('neg_sample_size = {}'.format(sampler_dict[r].neg_sample_size))

                for pos_g, neg_g in sampler_dict[r]:
                    model.forward_test(pos_g, neg_g, logs, gpu_id)

                print('len(logs) = {}'.format(len(logs) - log_len))
                log_len = len(logs)

        metrics = {}
        print('len(logs) = {}'.format(len(logs)))
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        if queue is not None:
            queue.put(logs)
        else:
            for k, v in metrics.items():
                print('[{}]{} average {}: {}'.format(rank, mode, k, v))

    for r in test_samplers[0]:
        test_samplers[0][r].reset()

    for r in test_samplers[1]:
        test_samplers[1][r].reset()

    # test_samplers[0] = test_samplers[0].reset()
    # test_samplers[1] = test_samplers[1].reset()

def test_rel_constrain_demo(args, model, test_samplers, id2entity_map, id2relation_map, wiki_name, rank=0, mode='Test', queue=None):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

    with th.no_grad():
        logs = []
        
        for sampler_dict in test_samplers: # head or tail sampler dict
            log_len = 0
            for r in sampler_dict:
                print(type(sampler_dict[r]))
                print('r = {}'.format(r))
                print('sampler mode = {}'.format(sampler_dict[r].mode))
                print('sampler neg_head = {}'.format(sampler_dict[r].neg_head))
                print('sampler probs = {}'.format(sampler_dict[r].probs))
                print('np.nonzero sampler_dict[r].probs = {}'.format(np.nonzero(sampler_dict[r].probs)))
                print('len(np.nonzero sampler_dict[r].probs) = {}'.format(np.nonzero(sampler_dict[r].probs).shape))
                print('neg_sample_size = {}'.format(sampler_dict[r].neg_sample_size))

                for pos_g, neg_g in sampler_dict[r]:
                    print('flag 1')
                    model.forward_test_demo(pos_g, neg_g, id2entity_map, id2relation_map, wiki_name, logs, gpu_id)
                
                print('len(logs) = {}'.format(len(logs) - log_len))
                log_len = len(logs)
        
        # print([np.nonzero(sampler_dict[r].probs).shape[0] for sampler_dict in test_samplers for r in sampler_dict])

        metrics = {}
        print('len(logs) = {}'.format(len(logs)))
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        if queue is not None:
            queue.put(logs)
        else:
            for k, v in metrics.items():
                print('[{}]{} average {}: {}'.format(rank, mode, k, v))

    for r in test_samplers[0]:
        test_samplers[0][r].reset()

    if len(test_samplers)>1:
        for r in test_samplers[1]:
            test_samplers[1][r].reset()

def test_demo(args, model, test_samplers, id2entity_map, id2relation_map, wiki_name, rank=0, mode='Test', queue=None):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

    with th.no_grad():
        logs = []
        # i = 0
        for sampler in test_samplers:
            # j = 0
            for pos_g, neg_g in sampler:
                model.forward_test_demo(pos_g, neg_g, id2entity_map, id2relation_map, wiki_name, logs, gpu_id)
                # j += 1
                # if j>1:
                    # break
            # i += 1
            # if i > 1:
                # sys.exit(0)

        metrics = {}
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        if queue is not None:
            queue.put(logs)
        else:
            for k, v in metrics.items():
                print('[{}]{} average {}: {}'.format(rank, mode, k, v))
    test_samplers[0] = test_samplers[0].reset()
    test_samplers[1] = test_samplers[1].reset()
    
@thread_wrapped_func
def train_mp_ke(args, iter_id, reg_loss_start_epoch, use_input_embedding, model, sg_model, log_queue, sg_optimizer, proj_layer_optimizer, id2entity_map, wiki_link_dict, train_sampler, valid_samplers, rank=0, rel_parts=None, cross_rels=None, barrier=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    train_ke(args, iter_id, reg_loss_start_epoch, use_input_embedding, model, sg_model, log_queue, sg_optimizer, proj_layer_optimizer, id2entity_map, wiki_link_dict, train_sampler, valid_samplers, rank, rel_parts, cross_rels, barrier)
    # print('end of train_mp_ke')
    # print(model.proj_layer.bias)

@thread_wrapped_func
def test_mp_demo(args, model, test_samplers, id2entity_map, id2relation_map, rank=0, mode='Test', queue=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    test_demo(args, model, test_samplers, id2entity_map, id2relation_map, rank, mode, queue)
    
@thread_wrapped_func
def test_mp(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    test(args, model, test_samplers, rank, mode, queue)
