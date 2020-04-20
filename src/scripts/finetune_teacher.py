"""
LoBERT: Low rank factorization for BERT

Authors:
 - Bumjoon Park (qkrskaqja@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: src/scripts/finetune_teacher.py
 - Scripts for simple fine tuning BERT-base for each task as a teacher.

Run NLI_KD_training.py with given settings
:argu task: choose within GLUE tasks, RACE 
:argu bert_model: only bert-base-uncased for LoBERT
:argu student_hidden_layers: 12 same as n_layer for BERT-base
:argu lr: fix same as BERT papar, 2e-5

You can ignore all the kd related arguments by setting alpha = 0, teacher_prediction = None.
********************************************
:argu kd_model: kd
:argu alpha: 0
:argu teacher_prediction: default, None
:argu beta: 0
:argu T: 0
********************************************

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
from multiprocessing import Pool

src_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_FOLDER)

from envs import HOME_DATA_FOLDER
from utils.util import run_process


# task = 'CoLA,SST-2,MRPC,QQP,MNLI,QNLI,RTE'

task = 'MRPC'
bert_model = 'bert-base-uncased'

assert bert_model in ['bert-base-uncased', 'bert-large-uncased'], 'bert models needs to be bert-base-uncased or bert-large-uncased'
n_layer = 12 if 'base' in bert_model else 24

run = 1
# all_lr = [1e-6, 5e-5, 2e-5]
# all_lr = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
# for fast test
all_lr = [2e-5]

all_cmds = collections.defaultdict(list)
n_gpu = torch.cuda.device_count()
cur_gpu = 0

assert n_gpu > 0

logging.info('will run {} for {} runs'.format(task, run))
logging.info('will run on %d GPUs' % n_gpu)

logging.info('all lr = {}'.format(all_lr))

if 'race' in task:
    pass
else:
    tasks = task.split(',')
    for t in tasks:
        for _ in range(int(run)):
            for lr in all_lr:
                cmd = 'python %s/NLI_KD_training.py ' % src_FOLDER
                options = ['--learning_rate', str(lr)]
                options += [
                    '--task_name', t,
                    '--alpha', '0.0',
                    '--T', '10.0',
                    '--bert_model', bert_model,
                    '--train_batch_size', '32',
                    '--eval_batch_size', '32',
                    '--output_dir', os.path.join(HOME_DATA_FOLDER, f'outputs/KD/{t}/teacher_{n_layer}layer'),
                    '--do_train', 'True',
                    '--do_eval', 'True',
                    '--beta', '0.0',
                    '--max_seq_length', '128',
                    '--fp16', 'False',
                    '--num_train_epochs', '4.0',  #At least 4 epochs ##piaotairen##
                    '--kd_model', 'kd',
                    '--log_every_step', '1',
                    '--gradient_accumulation_steps', '1',
                    '--student_hidden_layers', '12',
                ]

                cmd += ' '.join(options)
                all_cmds[cur_gpu].append('CUDA_VISIBLE_DEVICES=%d ' % cur_gpu + cmd)
                cur_gpu += 1
#                 You can skip GPU which is not available for run out memory problem. 
#                 if cur_gpu == 0:
#                     cur_gpu = 1

                cur_gpu %= n_gpu

run_cmd = [';'.join(all_cmds[k]) for k in all_cmds]

# print(run_cmd)
pool = Pool(processes=n_gpu)
pool.map(run_process, run_cmd)
