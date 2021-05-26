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

# ******************************
# NOTICE
# This file was modified by Vardaan Pahuja for this project (email: pahuja.9@osu.edu)
# ******************************

MAX_SENT_LEN = 1000
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
import multiprocessing_logging
import time
from functools import wraps

import dgl
from dgl.contrib import KVClient
import dgl.backend as F

from dataloader import EvalDataset, NewBidirectionalOneShotIterator
from dataloader import get_dataset
from tqdm import tqdm
import queue
import multiprocessing
import torch
import random
import signal

# cython module import
from wiki2vec.wikipedia2vec import train_page_custom
from wiki2vec.dictionary import Dictionary
from wiki2vec.mention_db import MentionDB
from wikipedia2vec.wikipedia2vec.link_graph import LinkGraph

def row_saprse_adagrad(name, ID, data, target, lr):
    """Row-Sparse Adagrad update function
    """
    original_name = name[0:-6]
    state_sum = target[original_name+'_state-data-']
    grad_sum = (data * data).mean(1)
    state_sum.index_add_(0, ID, grad_sum)
    std = state_sum[ID]  # _sparse_mask
    std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
    tmp = (-lr * data / std_values)
    target[name].index_add_(0, ID, tmp)


class KGEClient(KVClient):
    """User-defined kvclient for DGL-KGE
    """
    def set_clr(self, learning_rate):
        """Set learning rate for Row-Sparse Adagrad updater
        """
        self._udf_push_param = learning_rate


    def set_udf_push(self, push_handler):
        """Set user-defined push
        """
        self._udf_push_handler = push_handler


    def set_local2global(self, l2g):
        """Set local2global mapping
        """
        self._l2g = l2g


    def get_local2global(self):
        """Get local2global mapping
        """
        return self._l2g

def connect_to_kvstore(args, entity_pb, relation_pb, l2g):
    """Create kvclient and connect to kvstore service
    """
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    my_client = KGEClient(server_namebook=server_namebook)

    my_client.set_clr(args.lr)

    my_client.set_udf_push(row_saprse_adagrad)

    my_client.connect()

    if my_client.get_id() % args.num_client == 0:
        my_client.set_partition_book(name='entity_emb', partition_book=entity_pb)
        my_client.set_partition_book(name='relation_emb', partition_book=relation_pb)
    else:
        my_client.set_partition_book(name='entity_emb')
        my_client.set_partition_book(name='relation_emb')

    my_client.set_local2global(l2g)

    return my_client

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

# @profile
def train_process_sent_producer(p_id, titles_queue, data_queue, dump_db, link_indices, tokenizer, word_list, sample_ints, entity_list, word_count_actual, freq, freq_entities, args):
    if args.negative > 0:
        print('train_words = {}'.format(args.train_words))
        table_ptr_val = data_producer.init_unigram_table(word_list, freq, args.train_words)
        table_ptr_val_entities = data_producer.init_unigram_table(entity_list, freq_entities, args.train_entities)

    dictionary = Dictionary.load(args.dictionary_file)

    if args.link_graph_file is not None:
        link_graph = LinkGraph.load(args.link_graph_file, dictionary)
    else:
        link_graph = None

    mention_db = MentionDB.load(args.mention_db_file, dictionary)
    # title_list = list(dump_db.titles())

    # title_start_idx = len(title_list) * p_id // args.num_proc_sg # set start word of process to particular postion acc. to division

    batch_count = 0
    batch_placeholder = np.zeros((args.sg_batch_size, 2+2*args.negative), 'int64')

    # title_idx = title_start_idx

    # print('file_pos = {}'.format(file_pos))
    # train_file.seek(file_pos, 0) # reset file position to beginning of file

    last_word_cnt = 0
    word_cnt = 0

    sentence = []
    word_entity_tuples_list = []
    entity_word_tuples_list = []

    prev = ''
    eof = False

    # n_title_per_process = math.ceil(dump_db.page_size()/args.num_proc_sg/10) # reduce scale by a factor of 10
    # title_iterator = itertools.islice(titles_queue, p_id*n_title_per_process, (p_id+1)*n_title_per_process)
    
    while True:
        
        # if title_id>0:
            # break
        # if eof or train_file.tell() > file_pos + args.file_size / args.processes:
            # break
        # if title_idx > title_start_idx + len(title_list) // args.num_proc_sg - 1:
            # break

        try:
            title_id, title = titles_queue.get(block=False)
        except queue.Empty:
            break

        is_first_sentence = True
        # print('title = {}'.format(title))
        # train using Wikipedia link graph
        
        if args.link_graph_file is not None:
            entity_entity_tuples = []
            start = title_id * args.entities_per_page

            for i in range(start, start + args.entities_per_page):
                entity = link_indices[i]
                # print(entity)
                neighbors = link_graph.neighbor_indices(entity)
                # print(neighbors)
                for j in range(len(neighbors)):
                    entity_entity_tuples.append((entity, neighbors[j]))

            next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)
            chunk_0 = data_producer.sg_producer_entity_entity(entity_entity_tuples, table_ptr_val_entities, args.negative, next_random) # word-entity co-occurrences
        
        paragraphs = dump_db.get_paragraphs(title)

        # compute entity indices
        
        entity_indices = set()

        entity_indices.add(dictionary.get_entity_index(title))

        for paragraph in paragraphs:
            for wiki_link in paragraph.wiki_links:
                entity_indices.add(dictionary.get_entity_index(wiki_link.title))

        entity_indices.discard(-1)
        
        # print('flag process start')
        # word_entity_tuples, entity_word_tuples = train_page_custom(title, args.window, dump_db, dictionary, mention_db, tokenizer)
        # print('flag process finish')

        for para_id, paragraph in enumerate(paragraphs):
            text = paragraph.text
            text_len = len(text)
            tokens = tokenizer.tokenize(text)
            token_len = len(tokens)

            sentence.extend(list(filter(lambda x:x in word_list, map(lambda x:x.text.lower(), tokens))))

            # sample word-entity and entity-word co-occurrences here
            # print('paragraph = {}'.format(paragraph.text))
            if token_len>0:
                word_entity_tuples, entity_word_tuples = train_page_custom(paragraph, entity_indices, args.window, dump_db, dictionary, mention_db, tokenizer, sample_ints)

                word_entity_tuples_list.extend(word_entity_tuples)
                entity_word_tuples_list.extend(entity_word_tuples)

            if len(sentence) > 0 and (len(sentence)>=MAX_SENT_LEN or para_id==len(paragraphs)-1):
                # print('len(sentence) = {}'.format(len(sentence)))
                # subsampling
                # print('sentence = {}'.format(sentence))

                # sample word-word co-occurrences here

                sent_id = []
                if args.sample != 0:
                    sent_len = len(sentence)
                    i = 0
                    while i < sent_len:
                        word = sentence[i]
                        # print('flag 2')
                        # f = freq[word] / dictionary.word_size
                        # pb = (np.sqrt(f / args.sample) + 1) * args.sample / f;
                        pb = sample_ints[dictionary.get_word_index(word)]

                        if pb > np.random.random_sample():
                            sent_id.append(dictionary.get_word_index(word))
                        # else:
                            # print('word = {}'.format(word))
                            # print('word_list = {}'.format(word_list))
                        i += 1

                # print('len(sent_id) = {}'.format(len(sent_id)))
                
                if len(sent_id) < 2:
                    word_cnt += len(sentence)
                    sentence.clear()
                    entity_word_tuples_list.clear()
                    word_entity_tuples_list.clear()
                    continue
                

                next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

                vocab_size = dictionary.word_size
                
                # print('len(sent_id) = {}'.format(len(sent_id)))

                chunk_1 = data_producer.sg_producer(sent_id, len(sent_id), table_ptr_val, args.window, args.negative, vocab_size, args.sg_batch_size, next_random) # word-word co-occurrences
                
                next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

                chunk_2 = data_producer.sg_producer_word_entity(word_entity_tuples_list, table_ptr_val_entities, args.negative, next_random) # word-entity co-occurrences

                next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)
                
                chunk_3 = data_producer.sg_producer_entity_word(entity_word_tuples_list, table_ptr_val, args.negative, next_random) # word-entity co-occurrences

                # print('chunk_1 = {}, chunk_2 = {}, chunk_3 = {}'.format(chunk_1.shape, chunk_2.shape, chunk_3.shape))
                
                
                if args.link_graph_file is not None:
                    if is_first_sentence:
                        chunk = np.concatenate((chunk_0, chunk_1, chunk_2, chunk_3), axis=0)
                    else:
                        chunk = np.concatenate((chunk_1, chunk_2, chunk_3), axis=0)
                else:
                    chunk = np.concatenate((chunk_1, chunk_2, chunk_3), axis=0)
                
                # chunk = chunk_1
                # print('chunk = {}'.format(chunk.shape))

                chunk_pos = 0
                while chunk_pos < chunk.shape[0]:
                    # print('chunk_pos = {}'.format(chunk_pos))
                    # print('batch_count = {}'.format(batch_count))

                    remain_space = args.sg_batch_size - batch_count
                    remain_chunk = chunk.shape[0] - chunk_pos

                    if remain_chunk < remain_space:
                        take_from_chunk = remain_chunk
                    else:
                        take_from_chunk = remain_space

                    batch_placeholder[batch_count:batch_count+take_from_chunk, :] = chunk[chunk_pos:chunk_pos+take_from_chunk, :]
                    batch_count += take_from_chunk

                    if batch_count == args.sg_batch_size:
                        # print('flag 1')
                        # print('batch_placeholder = {}'.format(batch_placeholder.shape))
                        # print('flag c')
                        data_queue.put(batch_placeholder)
                        # print('flag d')
                        batch_count = 0

                    chunk_pos += take_from_chunk

                word_cnt += len(sentence)
                # print('word_cnt = {}'.format(word_cnt))
                # print('last_word_cnt = {}'.format(last_word_cnt))

                if word_cnt - last_word_cnt > 10000:
                    with word_count_actual.get_lock():
                        word_count_actual.value += word_cnt - last_word_cnt
                    last_word_cnt = word_cnt
                sentence.clear()
                entity_word_tuples_list.clear()
                word_entity_tuples_list.clear()
                is_first_sentence = False

        # title_idx += 1
        # data_queue.put(None)
        # print('word_cnt = {}'.format(word_cnt))
        # break

    with word_count_actual.get_lock():
        word_count_actual.value += word_cnt - last_word_cnt

    if batch_count > 0:
        data_queue.put(batch_placeholder[:batch_count,:])
    # data_queue.put(None)
    print('data producer {} complete'.format(p_id))


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

        # loss_combined = loss + args.balance_param*reg_loss

        log_queue.put("Loss/KE/{}: step id:{} = {}".format(str(rank), step, loss))

        print('initial loss = {}'.format(loss))
        if use_name_graph:
            loss_combined = args.balance_param * loss
        else:
            loss_combined = loss # in first epoch, no alignment takes place

        print('final loss = {}'.format(loss_combined))

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

def train_sg(pid, args, sg_model, titles_queue, data_queue, log_queue, link_indices, sg_optimizer, dump_db, word_count_actual, word2id_map, freq, word_list, sample_ints, entity2id_map, freq_entities, entity_list, tokenizer):
    # signal.signal(signal.SIGINT, signal_handler)
    #******************************************************************************
    # Train 1 epoch of SG model
    #******************************************************************************
    prev_word_cnt = 0
    vars(args)['t_start'] = time.monotonic()

    batch_id = 0

    while True:
        # print('flag 1')
        try:
            d = data_queue.get(block=True, timeout=args.timeout)
        except:
            break

        # print('word_count_actual.value = {}'.format(word_count_actual.value))
        # print('prev_word_cnt = {}'.format(prev_word_cnt))

        if word_count_actual.value - prev_word_cnt > 10000:
            lr = args.sg_lr * (1 - word_count_actual.value / (args.n_iters * args.train_words))
            if lr < 0.0001 * args.sg_lr:
                lr = 0.0001 * args.sg_lr
            for param_group in sg_optimizer.param_groups:
                param_group['lr'] = lr

            sys.stdout.write("\rpid:%d, Alpha: %0.8f, Progress: %0.2f, Words/sec: %f, Batches/sec: %f" % (pid, lr, word_count_actual.value / (args.n_iters * args.train_words) * 100, word_count_actual.value / (time.monotonic() - args.t_start), batch_id / (time.monotonic() - args.t_start)))
            sys.stdout.flush()
            # print('batch_id = {}'.format(batch_id))
            prev_word_cnt = word_count_actual.value

        sg_optimizer.zero_grad()
        batch_placeholder = th.LongTensor(d)

        if args.cuda:
            batch_placeholder = batch_placeholder.cuda()
            
        pos_loss, neg_loss = sg_model(batch_placeholder)
        sg_loss = pos_loss + neg_loss

        # print('sg grad = {}'.format(sg_model.emb0_lookup.weight.grad))
        log_queue.put("Loss/skip-gram/{}: step id:{} = {}".format(str(pid), batch_id, sg_loss))
        sg_loss.backward()
        # print('sg grad = {}'.format(sg_model.emb0_lookup.weight.grad))
        # print('training sg model')

        # print(torch.isnan(sg_model.emb0_lookup.weight.sum()).item())
        sg_optimizer.step()
        batch_id += 1

    #******************************************************************************

    # force synchronize embedding across processes every X steps
    
    # if args.force_sync_interval > 0 and (step + 1) % args.force_sync_interval == 0:
    #     barrier.wait()


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

@thread_wrapped_func
def train_mp_ke(args, iter_id, use_name_graph, reg_loss_start_epoch, use_input_embedding, model, sg_model, log_queue, sg_optimizer, proj_layer_optimizer, id2entity_map, wiki_link_dict, train_sampler, valid_samplers, rank=0, rel_parts=None, cross_rels=None, barrier=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    train_ke(args, iter_id, use_name_graph, reg_loss_start_epoch, use_input_embedding, model, sg_model, log_queue, sg_optimizer, proj_layer_optimizer, id2entity_map, wiki_link_dict, train_sampler, valid_samplers, rank, rel_parts, cross_rels, barrier)
    # print('end of train_mp_ke')
    # print(model.proj_layer.bias)

@thread_wrapped_func
def train_mp_sg(pid, args, sg_model, titles_queue, data_queue, log_queue, link_indices, sg_optimizer, dump_db, word_count_actual, word2id_map, freq, word_list, sample_ints, entity2id_map, freq_entities, entity_list, tokenizer):
    if args.num_proc_sg > 1:
        th.set_num_threads(args.num_thread)
    train_sg(pid, args, sg_model, titles_queue, data_queue, log_queue, link_indices, sg_optimizer, dump_db, word_count_actual, word2id_map, freq, word_list, sample_ints, entity2id_map, freq_entities, entity_list, tokenizer)

@thread_wrapped_func
def test_mp(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    test(args, model, test_samplers, rank, mode, queue)

@thread_wrapped_func
def dist_train_test(args, model, train_sampler, entity_pb, relation_pb, l2g, rank=0, rel_parts=None, cross_rels=None, barrier=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)

    client = connect_to_kvstore(args, entity_pb, relation_pb, l2g)
    client.barrier()
    train_time_start = time.time()
    train(args, model, train_sampler, None, rank, rel_parts, cross_rels, barrier, client)
    total_train_time = time.time() - train_time_start
    client.barrier()

    # Release the memory of local model
    model = None

    if (client.get_machine_id() == 0) and (client.get_id() % args.num_client == 0): # pull full model from kvstore
        # Pull model from kvstore
        args.num_test_proc = args.num_client
        dataset_full = dataset = get_dataset(args.data_path, args.dataset, args.format, args.data_files)
        args.train = False
        args.valid = False
        args.test = True
        args.strict_rel_part = False
        args.soft_rel_part = False
        args.async_update = False

        args.eval_filter = not args.no_eval_filter
        if args.neg_deg_sample_eval:
            assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

        print('Full data n_entities: ' + str(dataset_full.n_entities))
        print("Full data n_relations: " + str(dataset_full.n_relations))

        eval_dataset = EvalDataset(dataset_full, args)

        if args.neg_sample_size_eval < 0:
            args.neg_sample_size_eval = dataset_full.n_entities
        args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)

        model_test = load_model(None, args, dataset_full.n_entities, dataset_full.n_relations)

        print("Pull relation_emb ...")
        relation_id = F.arange(0, model_test.n_relations)
        relation_data = client.pull(name='relation_emb', id_tensor=relation_id)
        model_test.relation_emb.emb[relation_id] = relation_data
 
        print("Pull entity_emb ... ")
        # split model into 100 small parts
        start = 0
        percent = 0
        entity_id = F.arange(0, model_test.n_entities)
        count = int(model_test.n_entities / 100)
        end = start + count
        while True:
            print("Pull model from kvstore: %d / 100 ..." % percent)
            if end >= model_test.n_entities:
                end = -1
            tmp_id = entity_id[start:end]
            entity_data = client.pull(name='entity_emb', id_tensor=tmp_id)
            model_test.entity_emb.emb[tmp_id] = entity_data
            if end == -1:
                break
            start = end
            end += count
            percent += 1
    
        if not args.no_save_emb:
            print("save model to %s ..." % args.save_path)
            save_model(args, model_test)

        print('Total train time {:.3f} seconds'.format(total_train_time))

        if args.test:
            model_test.share_memory()
            start = time.time()
            test_sampler_tails = []
            test_sampler_heads = []
            for i in range(args.num_test_proc):
                test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.eval_filter,
                                                                mode='chunk-head',
                                                                num_workers=args.num_workers,
                                                                rank=i, ranks=args.num_test_proc)
                test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.eval_filter,
                                                                mode='chunk-tail',
                                                                num_workers=args.num_workers,
                                                                rank=i, ranks=args.num_test_proc)
                test_sampler_heads.append(test_sampler_head)
                test_sampler_tails.append(test_sampler_tail)

            eval_dataset = None
            dataset_full = None

            print("Run test, test processes: %d" % args.num_test_proc)

            queue = mp.Queue(args.num_test_proc)
            procs = []
            for i in range(args.num_test_proc):
                proc = mp.Process(target=test_mp, args=(args,
                                                        model_test,
                                                        [test_sampler_heads[i], test_sampler_tails[i]],
                                                        i,
                                                        'Test',
                                                        queue))
                procs.append(proc)
                proc.start()

            total_metrics = {}
            metrics = {}
            logs = []
            for i in range(args.num_test_proc):
                log = queue.get()
                logs = logs + log
            
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

            print("-------------- Test result --------------")
            for k, v in metrics.items():
                print('Test average {} : {}'.format(k, v))
            print("-----------------------------------------")

            for proc in procs:
                proc.join()

            print('testing takes {:.3f} seconds'.format(time.time() - start))

        client.shut_down() # shut down kvserver
