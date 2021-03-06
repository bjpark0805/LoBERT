"""
LoBERT: Low rank factorization for BERT

Authors:
 - Bumjoon Park (qkrskaqja@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: src/scripts/train_student.py
 - Scripts for training student model.
 
Run NLI_KD_training.py with given settings
:argu kd_model: 'lobert'
:argu teacher_prediction: saved teacher prediction
:argu lobert_mode: factorization method
:argu alpha, T: load the values from result_file
:argu beta, lr: grid search for the best hyperparameter 


Version: 1.0

Refer source code from https://github.com/intersun/PKD-for-BERT-Model-Compression.

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

"""

import os
import sys
import collections
import torch
import logging
import itertools
import pickle

import pandas as pd
from multiprocessing import Pool

src_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_FOLDER)

from envs import HOME_DATA_FOLDER, PREDICTION_FOLDER, PROJECT_FOLDER
from utils.util import run_process


result_file = {'3': os.path.join(PROJECT_FOLDER, 'src/result/glue/result_summary/kd_3layer_mean.csv'),
               '6': os.path.join(PROJECT_FOLDER, 'src/result/glue/result_summary/kd_6layer_mean.csv'),
               '12': os.path.join(PROJECT_FOLDER, 'src/result/glue/result_summary/kd_12layer_mean.csv')
               }

# task = 'MRPC,SST-2,MNLI,RTE'
# task = 'QQP,QNLI'

task = sys.argv[1]

student_layer = '12'
para_select = 'best'
teacher_layer = '12'
tasks = task.split(',')

# student_mode = 'svd, tensorlayer, qkv, falcon'
student_mode = sys.argv[2]


if teacher_layer == '12':
    bert_model = 'bert-base-uncased'
elif teacher_layer == '24':
    bert_model = 'bert-large-uncased'
else:
    raise ValueError(f'teacher_layer must be in [12, 24], now teacher_layer = {teacher_layer}')


# results = pickle.load(open(result_file[student_layer], 'rb'))['best_res_mean']
results = pd.read_csv(result_file[student_layer])

run = 1

if para_select == 'grid':
    all_T = {t: [5, 10, 20] for t in tasks}
    all_alpha = {t: [0.2, 0.5, 0.7] for t in tasks}
elif para_select == 'best':
    all_T, all_alpha = {}, {}
    for i, r in results.iterrows():
        tmp = r['uname'].split('_')
        all_T[tmp[1]] = [float('.'.join(tmp[4].split('.')[1:]))]
        all_alpha[tmp[1]] = [float('.'.join(tmp[5].split('.')[1:]))]
else:
    raise NotImplementedError(f'{para_select} not implemented')

all_lr = [5e-5, 2e-5, 1e-5]
# all_lr = [2e-5]
all_beta = [10, 100, 500, 1000]
# all_beta = [100]
normal_feat = ['True', 'False']

if teacher_layer == '12':
    if student_layer == '3':
        layer_idx = '9,10'
    elif student_layer == '6':
        layer_idx = '1,3,5,7,9'
    elif student_layer == '12':
        layer_idx = '0,1,2,3,4,5,6,7,8,9,10'
    else:
        raise ValueError(f'{student_layer} number of layers not supported yet')
elif teacher_layer == '24':
    if student_layer == '3':
        layer_idx = '1,3'
    elif student_layer == '6':
        layer_idx = '0,1,2,3,4'
    else:
        raise ValueError(f'{student_layer} number of layers not supported yet')

all_cmds = collections.defaultdict(list)
n_gpu = torch.cuda.device_count()
cur_gpu = 0

logging.info('will run {} for {} runs'.format(task, run))
logging.info('will run on %d GPUs' % n_gpu)

logging.info('all lr = {}'.format(all_lr))
logging.info('all T = {}'.format(all_T))
logging.info('all alpha = {}'.format(all_alpha))
logging.info('all beta = {}'.format(all_beta))
logging.info('layer_idx = {}'.format(layer_idx))
logging.info('lobert_mode = {}'.format(student_mode))

if 'race' in task:
    train_bs = 32
    eval_bs = 16
    gradient_accumulate = 4
    max_seq_length = 512
    do_eval = 'True'
else:
    train_bs = 32
    eval_bs = 32
    gradient_accumulate = 1
    max_seq_length = 128
    do_eval = 'False'

n_command = 0

for t in tasks:
    for _ in range(int(run)):
        for alpha, T, lr, beta, norm in itertools.product(all_alpha[t], all_T[t], all_lr, all_beta, normal_feat):
            cmd = 'python %s/src/NLI_KD_training.py ' % PROJECT_FOLDER
            options = [
                '--learning_rate', str(lr),
                '--alpha', str(alpha),
                '--T', str(T),
                '--beta', str(beta),
                '--normalize_patience', str(norm),
            ]
            options += [
                '--task_name', t,
                '--bert_model', bert_model,
                '--train_batch_size', str(train_bs),
                '--eval_batch_size', str(eval_bs),
                '--gradient_accumulation_steps', str(gradient_accumulate),
                '--max_seq_length', str(max_seq_length),
                '--do_eval',  str(do_eval),
                '--do_train', 'True',
                '--fp16', 'False',
                '--num_train_epochs', '4.0',
                '--kd_model', 'lobert',
                '--log_every_step', '10',
                '--student_hidden_layers', str(student_layer),
                '--fc_layer_idx', layer_idx,
                '--lobert_mode', student_mode,
            ]
            if beta > 0:
                options += [
                    '--teacher_prediction', os.path.join(HOME_DATA_FOLDER, f'outputs/LOBERT/{t}/{t}_lobert_teacher_{teacher_layer}layer_result_summary.pkl'),
                    '--output_dir', os.path.join(HOME_DATA_FOLDER, f'outputs/LOBERT/{t}/lobert_{student_mode}_teacher{teacher_layer}'),
                    ]
            else:
                raise ValueError('for patience teacher beta needs to be large than 0')

            cmd += ' '.join(options)
            all_cmds[cur_gpu].append('CUDA_VISIBLE_DEVICES=%d ' % cur_gpu + cmd)
            cur_gpu += 1
            cur_gpu %= n_gpu
#             You can skip GPU which is not available for run out memory problem. 
#             if cur_gpu == 0:
#                 cur_gpu = 1
         
            n_command += 1

run_cmd = [';'.join(all_cmds[k]) for k in all_cmds]
logging.info(f'will run {n_command} number of commands')

pool = Pool(processes=n_gpu)
pool.map(run_process, run_cmd)