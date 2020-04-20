"""
LoBERT: Low rank factorization for BERT

Authors:
 - Bumjoon Park (qkrskaqja@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: src/scripts/save_teacher_prediction.py
 - Scripts for saving teacher model's prediction.

:argu[1]: tasks, choose within GLUE tasks
:argu[2]: lobert mode for output_directory, whether to save predictions for train, dev
:argu[3]: whether to save predictions for all layers, true
:argu[4]: result_file, for finding teacher model directory
:argu[5]: bert_model, only bert-base-uncased for LoBERT

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
from multiprocessing import Pool

src_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_FOLDER)
from envs import HOME_DATA_FOLDER, PREDICTION_FOLDER
from utils.util import run_process
# task = 'SST-2,MRPC,QQP,QNLI,RTE,MNLI'
task = 'MRPC'
bert_model = 'bert-base-uncased'
assert bert_model in ['bert-base-uncased', 'bert-large-uncased'], 'bert models needs to be bert-base-uncased or bert-large-uncased'
n_layer = 12 if 'base' in bert_model else 24


all_cmds = collections.defaultdict(list)
n_gpu = torch.cuda.device_count()
cur_gpu = 0

logging.info('will run on %d GPUs' % n_gpu)
tasks = task.split(',')
for t in tasks:
    cmd = f'python {src_FOLDER}/run_glue_benchmark.py {t} lobert:train,dev True '
    cmd += os.path.join(src_FOLDER, 'result/glue/result_summary/teacher_12layer_all.csv')
    cmd += ' ' + bert_model
    all_cmds[cur_gpu].append('CUDA_VISIBLE_DEVICES=%d ' % cur_gpu + cmd)
    cur_gpu += 1
#     You can skip GPU which is not available for run out memory problem. 
#     if cur_gpu == 0:
#         cur_gpu = 1
    cur_gpu %= n_gpu

run_cmd = [';'.join(all_cmds[k]) for k in all_cmds]

# print(run_cmd)
pool = Pool(processes=n_gpu)
pool.map(run_process, run_cmd)