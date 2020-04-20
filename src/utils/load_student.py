"""
LoBERT: Low rank factorization for BERT

Authors:
 - Bumjoon Park (qkrskaqja@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: src/utils/load_student.py
 - Compress student model for LoBERT.

Version: 1.0


This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

"""


import logging
import torch
import os

import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD
from torch import nn
from tqdm import tqdm
from bert.pytorch_pretrained_bert.modeling import BertConfig, BertSelfAttention_svd, BertSelfAttention_tucker_layer, BertSelfAttention_FALCON_layer, BertEncoder_tucker_vertical, BertEncoder_FALCON_vertical, BertEncoder_tucker_4d, BertEncoder_FALCON_4d, BertEncoder_ffn_tucker, BertIntermediate_svd, BertOutput_svd
from utils.modeling import BertForSequenceClassificationEncoder, FCClassifierForSequenceClassification, FullFCClassifierForSequenceClassification
from tensorly.decomposition import tucker
import time
import tensorly as tl
tl.set_backend('pytorch')

logger = logging.getLogger(__name__)

def lobert_student(config, student_model, checkpoint, args, student_mode='exact', num_hidden_layers = 12, verbose=True, DEBUG=False):
    """
    Create and initiate student_model. 
    :param config: configuration of parent model
    :param student_model: student model
    :param checkpoint: checkpoint of parent model
    :param args: global argument
    :param student_mode: decide factorization method of student_model, 'exact' means not compressed and will raise error 
    :param num_hidden_layers : number of layers of student model
    :param verbose: explain error
    :param DEBUG: check deleted parameters of student_model
    :return: student_model
    """
    if student_mode == 'exact':
        raise NotImplementedError('not implemented for student mode')

    n_gpu = args.n_gpu
    device = args.device
    local_rank = -1
 
    if checkpoint in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % student_model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (student_model._get_name(), checkpoint))

        model_state_dict = torch.load(checkpoint)
        
        old_keys = []
        new_keys = []
        for key in model_state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            model_state_dict[new_key] = model_state_dict.pop(old_key)

        del_keys = []
        keep_keys = []
        
        model_keys = student_model.state_dict().keys()
        for t in list(model_state_dict.keys()):
            if t not in model_keys:
                del model_state_dict[t]
                del_keys.append(t)
            else:
                keep_keys.append(t)
        
        student_model.load_state_dict(model_state_dict)
  
        logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))
        if DEBUG:
            print('deleted keys =\n {}'.format('\n'.join(del_keys)))
            print('*' * 77)
            print('kept keys =\n {}'.format('\n'.join(keep_keys)))


    intermediate_tensor = []
    output_tensor = []
    
    self_attention_tensor = []
    layer_tensor = []
    for i in range(num_hidden_layers):
        intermediate_tensor.append(student_model.bert.encoder.layer[i].intermediate.dense.weight)
        output_tensor.append(student_model.bert.encoder.layer[i].output.dense.weight)

        layer_tensor = []
        layer_tensor.append(student_model.bert.encoder.layer[i].attention.self.query.weight.data)
        layer_tensor.append(student_model.bert.encoder.layer[i].attention.self.key.weight.data)
        layer_tensor.append(student_model.bert.encoder.layer[i].attention.self.value.weight.data)
        layer_tensor = torch.stack(layer_tensor)
        self_attention_tensor.append(layer_tensor)
    
    intermediate_tensor = torch.stack(intermediate_tensor)  # shape = [12, 3072, 768]
    output_tensor = torch.stack(output_tensor) # shape = [12, 768, 3072]

    self_attention_tensor = torch.stack(self_attention_tensor) # shape = [12, 3, 768, 768]
    
    if student_mode == 'self_svd':
        logger.info('lobert mode : self_SVD_compression begin')
        student_model = decompose_self_svd(student_model, self_attention_tensor, config, num_hidden_layers) 
    elif student_mode == 'ffn_svd':
        logger.info('lobert mode : ffn_SVD_compression begin')
        student_model = decompose_ffn_svd(student_model, intermediate_tensor, output_tensor, config, num_hidden_layers)
    elif student_mode == 'ffn_tuckervertical':
        logger.info('lobert mode : ffn_tucker_compression begin')
        student_model = decompose_ffn_tucker_vertical(student_model, intermediate_tensor, output_tensor, config, num_hidden_layers)
    elif student_mode == 'self_tuckerlayer':
        logger.info('lobert mode : self_tucker_layer compression begin')
        student_model = decompose_self_tucker_layer(student_model, self_attention_tensor, config, num_hidden_layers)
    elif student_mode == 'self_falconlayer':
        logger.info('lobert mode : self_falcon_layer compression begin')
        student_model = decompose_self_falcon_layer(student_model, self_attention_tensor, config, num_hidden_layers)
    elif student_mode == 'self_tuckervertical':
        logger.info('lobert mode : self_tucker_vertical compression begin')
        student_model = decompose_self_tucker_vertical(student_model, self_attention_tensor, config,num_hidden_layers)
    elif student_mode == 'self_falconvertical':
        logger.info('lobert mode : self_falcon_vertical compression begin')
        student_model = decompose_self_falcon_vertical(student_model, self_attention_tensor, config,num_hidden_layers)
    elif student_mode == 'self_tucker_4d':
        logger.info('lobert mode : self_tucker_4d compression begin')
        student_model = decompose_self_tucker_4d(student_model, self_attention_tensor, config,num_hidden_layers)
    elif student_mode == 'self_falcon_4d':
        logger.info('lobert mode : self_falcon_4d compression begin')
        student_model = decompose_self_falcon_4d(student_model, self_attention_tensor, config,num_hidden_layers)
    else:
        raise NotImplementedError('not implemented for this student mode')
            
            
    if args.fp16:
        logger.info('fp16 activated, now call model.half()')
        student_model.half()
    student_model.to(device)


    if local_rank != -1:
        raise NotImplementedError('not implemented for local_rank != 1')
    elif n_gpu > 1:
        logger.info('data parallel because more than one gpu')
        student_model = torch.nn.DataParallel(student_model)
    return student_model


def decompose_ffn_svd(student_model, intermediate_tensor, output_tensor, config, num_hidden_layers, svd_rank = 306):
    """
    SVD Decompose student_model for intermediate and output for all the layers. 
    :param student_model: student model
    :param intermediate_tensor: Tensor of intermediate matrix for all the layers.
    :param output_tensor: Tensor of output matrix for all the layers.
    :param config: configuration of parent model
    :param num_hidden_layers: number of layers of student model
    :param svd_rank: SVD decomposition rank
    :return: student_model
    """
    #     svd_rank : 51, 153, 306
    for i in range(config.num_hidden_layers):
        intermediate_weight = intermediate_tensor[i]
        U, S, V = tl.partial_svd(intermediate_weight, n_eigenvecs=svd_rank)
        S = torch.diag(S)
        U = U.matmul(S)
        intermediate = BertIntermediate_svd(config, svd_rank)
        intermediate.dense_u.weight.data.copy_(V)
        intermediate.dense_v.weight.data.copy_(U)
        intermediate.dense_v.bias.data.copy_(student_model.bert.encoder.layer[i].intermediate.dense.bias.data)
        
        output_weight = output_tensor[i]
        U, S, V = tl.partial_svd(output_weight, n_eigenvecs=svd_rank)
        S = torch.diag(S)
        U = U.matmul(S)
        output = BertOutput_svd(config, svd_rank)
        output.dense_u.weight.data.copy_(V)
        output.dense_v.weight.data.copy_(U)
        output.dense_v.bias.data.copy_(student_model.bert.encoder.layer[i].output.dense.bias.data)

        student_model.bert.encoder.layer[i].intermediate = intermediate
        student_model.bert.encoder.layer[i].output = output
        
    return student_model

def decompose_ffn_tucker_vertical(student_model, intermediate_tensor, output_tensor, config, num_hidden_layers, rank_fst = 8, rank_snd = 492):
    """
    Decompose student_model for intermediate and output layer with tucker decomposition. 
    :param student_model: student model
    :param intermediate_t: Tensor of intermediate matrix for all the layers.
    :param output_t: Tensor of output matrix for all the layers.
    :param config: configuration of parent model
    :param num_hidden_layers: number of layers of student model
    :param rank_fst, rank_snd: tucker decomposition rank (rank_fst = 8, rank_snd = 492 (50%), 309(25%), 136(8.3%)  
    :return: student_model
    """
#     rank_snd : 136, 309, 409
#     intermediate_tensor.shape = [12, 3072, 768]
#     output_tensor.shape = [12, 768, 3072]
    
    core_i, factors_i = tucker(intermediate_tensor, ranks=[rank_fst, rank_snd*4, rank_snd]) 
    core_o, factors_o = tucker(output_tensor, ranks=[rank_fst, rank_snd, rank_snd*4])    
            
    encoder = BertEncoder_ffn_tucker(config, rank_fst, rank_snd)
    encoder.tensor_core_i.data.copy_(core_i)
    encoder.matrix_a_i.data.copy_(factors_i[0])
    encoder.matrix_b_i.data.copy_(factors_i[1])
    encoder.matrix_c_i.data.copy_(factors_i[2])
    
    encoder.tensor_core_o.data.copy_(core_o)
    encoder.matrix_a_o.data.copy_(factors_o[0])
    encoder.matrix_b_o.data.copy_(factors_o[1])
    encoder.matrix_c_o.data.copy_(factors_o[2])
    
    for i in range(config.num_hidden_layers):
        encoder.layer[i].intermediate.bias.data.copy_(student_model.bert.encoder.layer[i].intermediate.dense.bias.data)
        encoder.layer[i].output.bias.data.copy_(student_model.bert.encoder.layer[i].output.dense.bias.data)
        encoder.layer[i].attention = student_model.bert.encoder.layer[i].attention
        
    student_model.bert.encoder = encoder
        
    return student_model

def decompose_self_svd(student_model, self_attention_tensor, config, num_hidden_layers, svd_rank=192):
    """
    SVD Decompose student_model for Query, Key, Value of all the layers.
    :param student_model: student model
    :param self_attention_tensor: stack of stacked Query, Key, Value for all the layers
    :param config: configuration of parent model
    :param num_hidden_layers: number of layers of student model
    :param svd_rank: SVD decomposition rank
    :return: student_model
    """
    for i in range(num_hidden_layers):
        weight = self_attention_tensor[i]
        q = weight[0]
        k = weight[1]
        v = weight[2]
        Uq, Sq, Vq = tl.partial_svd(q, n_eigenvecs=svd_rank)
        Sq = torch.diag(Sq)
        Uq = Uq.matmul(Sq)
        Uk, Sk, Vk = tl.partial_svd(k, n_eigenvecs=svd_rank)
        Sk = torch.diag(Sk)
        Uk = Uk.matmul(Sk)
        Uv, Sv, Vv = tl.partial_svd(v, n_eigenvecs=svd_rank)
        Sv = torch.diag(Sv)
        Uv = Uv.matmul(Sv)
        attention_layer = BertSelfAttention_svd(config, svd_rank)
        attention_layer.query_u.weight.data.copy_(Vq)
        attention_layer.query_v.weight.data.copy_(Uq)
        attention_layer.query_v.bias.data.copy_(student_model.bert.encoder.layer[i].attention.self.query.bias.data)

        attention_layer.key_u.weight.data.copy_(Vk)
        attention_layer.key_v.weight.data.copy_(Uk)
        attention_layer.key_v.bias.data.copy_(student_model.bert.encoder.layer[i].attention.self.value.bias.data)

        attention_layer.value_u.weight.data.copy_(Vv)
        attention_layer.value_v.weight.data.copy_(Uv)
        attention_layer.value_v.bias.data.copy_(student_model.bert.encoder.layer[i].attention.self.value.bias.data)

        student_model.bert.encoder.layer[i].attention.self = attention_layer
    
    return student_model


def decompose_self_tucker_layer(student_model, self_attention_tensor, config, num_hidden_layers, rank_fst=2, rank_snd=384):
    """
    Tucker decompose student_model for tensor of stacked Query, Key, Value of each layer. 
    :param student_model: student model
    :param self_attention_tensor: stack of stacked Query, Key, Value for all the layers
    :param config: configuration of parent model
    :param num_hidden_layers: number of layers of student model
    :param rank_fst: Tucker decomposition rank of factor 1
    :param rank_snd: Tucker decomposition rank of factor 2 & 3
    :return: student_model
    """
    for i in range(config.num_hidden_layers):
        weight = self_attention_tensor[i] # shape : [3, 768, 768]
        core, factors = tucker(weight, ranks=[rank_fst, rank_snd, rank_snd])
        
        layer = BertSelfAttention_tucker_layer(config, rank_fst, rank_snd)
        layer.tensor_core.data.copy_(core)
        layer.matrix_a.data.copy_(factors[0])
        layer.matrix_b.data.copy_(factors[1])
        layer.matrix_c.data.copy_(factors[2])
        layer.bias_q.data.copy_(student_model.bert.encoder.layer[i].attention.self.query.bias.data)
        layer.bias_k.data.copy_(student_model.bert.encoder.layer[i].attention.self.key.bias.data)
        layer.bias_v.data.copy_(student_model.bert.encoder.layer[i].attention.self.value.bias.data)
        
        student_model.bert.encoder.layer[i].attention.self = layer

    return student_model

def decompose_self_falcon_layer(student_model, self_attention_tensor, config, num_hidden_layers):
    """
    FALCON decompose student_model for tensor of stacked Query, Key, Value of each layer. 
    :param student_model: student model
    :param self_attention_tensor: stack of stacked Query, Key, Value for all the layers
    :param config: configuration of parent model
    :param num_hidden_layers: number of layers of student model
    :param rank_fst: Tucker decomposition rank of factor 1
    :param rank_snd: Tucker decomposition rank of factor 2 & 3
    :return: student_model
    """
    
    for i in range(config.num_hidden_layers):
        weight = self_attention_tensor[i] # shape : [3, 768, 768]
        lr=0.01
        steps=10000
        dw = torch.randn([3,768,1],requires_grad=True)
        pw = torch.randn([1,768,768],requires_grad=True)

        weight = torch.FloatTensor(weight)
        criterion = nn.MSELoss()
        optimizer = SGD({pw, dw}, lr=lr)
        st = time.time()
        for s in range(steps):
            if s == 4000 or s == 7000:
                lr = lr / 10
                optimizer = SGD({pw, dw}, lr=lr)
            optimizer.zero_grad()
            kernel_pred = pw.cuda() * dw.cuda()
            loss = criterion(kernel_pred, weight.cuda())
            loss.backward()
            optimizer.step()
            if s % 10000 == 9999:
                print('loss = %f, time = %d' % (loss, (time.time() - st)))
                st = time.time()
        
        layer = BertSelfAttention_FALCON_layer(config)
        layer.dw.data.copy_(dw)
        layer.pw.data.copy_(pw)
        layer.bias_q.data.copy_(student_model.bert.encoder.layer[i].attention.self.query.bias.data)
        layer.bias_k.data.copy_(student_model.bert.encoder.layer[i].attention.self.key.bias.data)
        layer.bias_v.data.copy_(student_model.bert.encoder.layer[i].attention.self.value.bias.data)
        
        student_model.bert.encoder.layer[i].attention.self = layer

    return student_model

def decompose_self_tucker_vertical(student_model, self_attention_tensor, config, num_hidden_layers, rank_fst=8, rank_snd=576):
    """
    Tucker decompose student_model for stacked Query for all the layers(Same in Key, Value). 
    :param student_model: student model
    :param self_attention_tensor: stack of stacked Query, Key, Value for all the layers
    :param config: configuration of parent model
    :param num_hidden_layers: number of layers of student model
    :param rank_fst: Tucker decomposition rank of factor 1
    :param rank_snd: Tucker decomposition rank of factor 2 & 3
    :return: student_model
    """
    self_attention_tensor = self_attention_tensor.permute(1, 0, 2, 3)   #shape = [3, 12, 768, 768]
    weight_q = self_attention_tensor[0]
    core_q, factors_q = tucker(weight_q, ranks=[rank_fst, rank_snd, rank_snd])
    weight_k = self_attention_tensor[1]
    core_k, factors_k = tucker(weight_k, ranks=[rank_fst, rank_snd, rank_snd])
    weight_v = self_attention_tensor[2]
    core_v, factors_v = tucker(weight_v, ranks=[rank_fst, rank_snd, rank_snd])
    
    encoder = BertEncoder_tucker_vertical(config, rank_fst, rank_snd)
    encoder.tensor_core_q.data.copy_(core_q)
    encoder.matrix_a_q.data.copy_(factors_q[0])
    encoder.matrix_b_q.data.copy_(factors_q[1])
    encoder.matrix_c_q.data.copy_(factors_q[2])

    encoder.tensor_core_k.data.copy_(core_k)
    encoder.matrix_a_k.data.copy_(factors_k[0])
    encoder.matrix_b_k.data.copy_(factors_k[1])
    encoder.matrix_c_k.data.copy_(factors_k[2])
    
    encoder.tensor_core_v.data.copy_(core_v)
    encoder.matrix_a_v.data.copy_(factors_v[0])
    encoder.matrix_b_v.data.copy_(factors_v[1])
    encoder.matrix_c_v.data.copy_(factors_v[2])
    
    for i in range(config.num_hidden_layers):
        encoder.layer[i].attention.self.bias_q.data.copy_(student_model.bert.encoder.layer[i].attention.self.query.bias.data)
        encoder.layer[i].attention.self.bias_k.data.copy_(student_model.bert.encoder.layer[i].attention.self.key.bias.data)
        encoder.layer[i].attention.self.bias_v.data.copy_(student_model.bert.encoder.layer[i].attention.self.value.bias.data)
        encoder.layer[i].intermediate = student_model.bert.encoder.layer[i].intermediate
        encoder.layer[i].output = student_model.bert.encoder.layer[i].output
        encoder.layer[i].attention.output = student_model.bert.encoder.layer[i].attention.output
        
    student_model.bert.encoder = encoder
    
    return student_model

def decompose_self_falcon_vertical(student_model, self_attention_tensor, config, num_hidden_layers):
    """
    FALCON decompose student_model for stacked Query for all the layers(Same in Key, Value). 
    :param student_model: student model
    :param self_attention_tensor: stack of stacked Query, Key, Value for all the layers
    :param config: configuration of parent model
    :param num_hidden_layers: number of layers of student model
    :param rank_fst: Tucker decomposition rank of factor 1
    :param rank_snd: Tucker decomposition rank of factor 2 & 3
    :return: student_model
    """
    
    self_attention_tensor = self_attention_tensor.permute(1, 0, 2, 3)   #shape = [3, 12, 768, 768]
    weight_q = self_attention_tensor[0] # shape = [12, 768, 768] 
    weight_k = self_attention_tensor[1] # shape = [12, 768, 768]
    weight_v = self_attention_tensor[2] # shape = [12, 768, 768]
    
    lr=0.01
    steps=10000
    dw_q = torch.randn([12,768,1],requires_grad=True)
    pw_q = torch.randn([1,768,768],requires_grad=True)
    
    dw_k = torch.randn([12,768,1],requires_grad=True)
    pw_k = torch.randn([1,768,768],requires_grad=True)
    
    dw_v = torch.randn([12,768,1],requires_grad=True)
    pw_v = torch.randn([1,768,768],requires_grad=True)
    
    weight_q = torch.FloatTensor(weight_q)
    weight_k = torch.FloatTensor(weight_k)
    weight_v = torch.FloatTensor(weight_v)
    
    criterion = nn.MSELoss()
    optimizer = SGD({pw_q, dw_q}, lr=lr)
    st = time.time()
    for s in range(steps):
        if s == 4000 or s == 7000:
            lr = lr / 10
            optimizer = SGD({pw_q, dw_q}, lr=lr)
        optimizer.zero_grad()
        kernel_pred = pw_q.cuda() * dw_q.cuda()
        loss = criterion(kernel_pred, weight_q.cuda())
        loss.backward()
        optimizer.step()
        if s % 10000 == 9999:
            print('loss = %f, time = %d' % (loss, (time.time() - st)))
            st = time.time()
            
    optimizer = SGD({pw_k, dw_k}, lr=lr)
    st = time.time()
    for s in range(steps):
        if s == 4000 or s == 7000:
            lr = lr / 10
            optimizer = SGD({pw_k, dw_k}, lr=lr)
        optimizer.zero_grad()
        kernel_pred = pw_k.cuda() * dw_k.cuda()
        loss = criterion(kernel_pred, weight_k.cuda())
        loss.backward()
        optimizer.step()
        if s % 10000 == 9999:
            print('loss = %f, time = %d' % (loss, (time.time() - st)))
            st = time.time()
            
    optimizer = SGD({pw_v, dw_v}, lr=lr)
    st = time.time()
    for s in range(steps):
        if s == 4000 or s == 7000:
            lr = lr / 10
            optimizer = SGD({pw_v, dw_v}, lr=lr)
        optimizer.zero_grad()
        kernel_pred = pw_v.cuda() * dw_v.cuda()
        loss = criterion(kernel_pred, weight_v.cuda())
        loss.backward()
        optimizer.step()
        if s % 10000 == 9999:
            print('loss = %f, time = %d' % (loss, (time.time() - st)))
            st = time.time()
                   

    encoder = BertEncoder_FALCON_vertical(config)
    encoder.dw_q.data.copy_(dw_q)
    encoder.pw_q.data.copy_(pw_q)
    encoder.dw_k.data.copy_(dw_k)
    encoder.pw_k.data.copy_(pw_k)
    encoder.dw_v.data.copy_(dw_v)
    encoder.pw_v.data.copy_(pw_v)
    
    for i in range(config.num_hidden_layers):
        encoder.layer[i].attention.self.bias_q.data.copy_(student_model.bert.encoder.layer[i].attention.self.query.bias.data)
        encoder.layer[i].attention.self.bias_k.data.copy_(student_model.bert.encoder.layer[i].attention.self.key.bias.data)
        encoder.layer[i].attention.self.bias_v.data.copy_(student_model.bert.encoder.layer[i].attention.self.value.bias.data)
        encoder.layer[i].intermediate = student_model.bert.encoder.layer[i].intermediate
        encoder.layer[i].output = student_model.bert.encoder.layer[i].output
        encoder.layer[i].attention.output = student_model.bert.encoder.layer[i].attention.output
        
    student_model.bert.encoder = encoder
    
    return student_model

def decompose_self_tucker_4d(student_model, self_attention_tensor, config, num_hidden_layers, rank_fst=2, rank_snd=8, rank_trd = 576):
    """
    Tucker decompose student_model for stack of stacked Query, Key, Value for all the layers. 
    :param student_model: student model
    :param self_attention_tensor: Tensor of stack of Query, Key, Value for all the layers.
    :param config: configuration of parent model
    :param num_hidden_layers: number of layers of student model
    :param rank_fst: Tucker decomposition rank of factor 1
    :param rank_snd: Tucker decomposition rank of factor 2
    :param rank_trd: Tucker decomposition rank of factor 3, 4
    :return: student_model
    """
    self_attention_tensor = self_attention_tensor.permute(1, 0, 2, 3) #shape = [3, 12, 768, 768]
    core, factors = tucker(self_attention_tensor, ranks=[rank_fst, rank_snd, rank_trd, rank_trd])
    
    encoder = BertEncoder_tucker_4d(config, rank_fst, rank_snd, rank_trd)
    encoder.core.data.copy_(core)
    encoder.matrix_a.data.copy_(factors[0])
    encoder.matrix_b.data.copy_(factors[1])
    encoder.matrix_c.data.copy_(factors[2])
    encoder.matrix_d.data.copy_(factors[3])
    
    for i in range(config.num_hidden_layers):
        encoder.layer[i].attention.self.bias_q.data.copy_(student_model.bert.encoder.layer[i].attention.self.query.bias.data)
        encoder.layer[i].attention.self.bias_k.data.copy_(student_model.bert.encoder.layer[i].attention.self.key.bias.data)
        encoder.layer[i].attention.self.bias_v.data.copy_(student_model.bert.encoder.layer[i].attention.self.value.bias.data)
        encoder.layer[i].intermediate = student_model.bert.encoder.layer[i].intermediate
        encoder.layer[i].output = student_model.bert.encoder.layer[i].output
        encoder.layer[i].attention.output = student_model.bert.encoder.layer[i].attention.output
 
    student_model.bert.encoder = encoder
    
    return student_model

def decompose_self_falcon_4d(student_model, self_attention_tensor, config, num_hidden_layers):
    """
    FALCON decompose student_model for stack of stacked Query, Key, Value for all the layers. 
    :param student_model: student model
    :param self_attention_tensor: Tensor of stack of Query, Key, Value for all the layers.
    :param config: configuration of parent model
    :param num_hidden_layers: number of layers of student model
    :return: student_model
    """
    self_attention_tensor = self_attention_tensor.permute(1, 0, 2, 3)
    lr=0.01
    steps=10000
    # dw's dimension: 1x12x768x768
    dw = torch.randn([1,12,768,768],requires_grad=True)
    # pw's dimension: 3x12x1x1
    pw = torch.randn([3,12,1,1],requires_grad=True)
    
    self_attention_tensor = torch.FloatTensor(self_attention_tensor)
    criterion = nn.MSELoss()
    optimizer = SGD({pw, dw}, lr=lr)
    st = time.time()
    for s in range(steps):
        if s == 4000 or s == 7000:
            lr = lr / 10
            optimizer = SGD({pw, dw}, lr=lr)
        optimizer.zero_grad()
        kernel_pred = pw.cuda() * dw.cuda()
        loss = criterion(kernel_pred, self_attention_tensor.cuda())
        loss.backward()
        optimizer.step()
        if s % 10000 == 9999:
            print('loss = %f, time = %d' % (loss, (time.time() - st)))
            st = time.time()
                   

    encoder = BertEncoder_FALCON_4d(config)
    encoder.dw.data.copy_(dw)
    encoder.pw.data.copy_(pw)
    for i in range(config.num_hidden_layers):
        encoder.layer[i].attention.self.bias_q.data.copy_(student_model.bert.encoder.layer[i].attention.self.query.bias.data)
        encoder.layer[i].attention.self.bias_k.data.copy_(student_model.bert.encoder.layer[i].attention.self.key.bias.data)
        encoder.layer[i].attention.self.bias_v.data.copy_(student_model.bert.encoder.layer[i].attention.self.value.bias.data)
        encoder.layer[i].intermediate = student_model.bert.encoder.layer[i].intermediate
        encoder.layer[i].output = student_model.bert.encoder.layer[i].output
        encoder.layer[i].attention.output = student_model.bert.encoder.layer[i].attention.output
 
    student_model.bert.encoder = encoder
    
    return student_model