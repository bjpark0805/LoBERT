"""
LoBERT: Low rank factorization for BERT

Authors:
 - Bumjoon Park (qkrskaqja@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: src/run_glue_benchmark.py
 - Test the model or save its prediction while testing.
 
Run run_glue_benchmark.py with given settings
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


import pickle
import os
import glob
import logging
import argparse
import torch
import sys

import pandas as pd
import numpy as np
from torch.utils.data import SequentialSampler

# PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PROJECT_FOLDER)

from utils import nli_data_processing
from envs import PROJECT_FOLDER, HOME_DATA_FOLDER
from bert.pytorch_pretrained_bert.modeling import BertConfig
from bert.pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.modeling import BertForSequenceClassificationEncoder, FCClassifierForSequenceClassification
from utils.util import count_parameters, load_model, eval_model_dataloader, fill_tensor
from utils.data_processing import init_model, get_task_dataloader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False

ALL_TASKS = ['MRPC', 'RTE', 'SST-2', 'MNLI', 'QQP', 'MNLI-mm', 'QNLI', 'race-merge']

lobert_mode = None
if DEBUG:
    interested_task = 'RTE'.split(',')
    prediction_mode_input = 'teacher:train,dev,test'
    output_all_layers = True   # True for patient teacher and False for normal teacher
    bert_model = 'bert-base-uncased'
    result_file = os.path.join(PROJECT_FOLDER, 'src/result/glue/result_summary/teacher_12layer_all.csv')
else:
    interested_task = sys.argv[1].split(',')
    prediction_mode_input = sys.argv[2]    
    output_all_layers = sys.argv[3].lower() == 'true'
    result_file = sys.argv[4]
    bert_model = sys.argv[5]
    if(len(sys.argv) >= 7):
        lobert_mode = sys.argv[6]

for t in interested_task:
    assert t in ALL_TASKS, f'{t} not in all tasks! double check!'

##############################################################
# Global Variables
##############################################################
AVAILABLE_MODE = ['teacher', 'benchmark','lobert','lobert_test']
KD_DIR = os.path.join(HOME_DATA_FOLDER, 'outputs/KD/')
LOBERT_DIR = os.path.join(HOME_DATA_FOLDER, 'outputs/LOBERT/')

sub_dir = '_'.join(os.path.basename(result_file).split('_')[:-1]) # ex) teacher_12layer

prediction_mode, interested_set = prediction_mode_input.split(':')
assert prediction_mode in AVAILABLE_MODE, f'mode {prediction_mode} not available'


if prediction_mode == 'teacher':
    output_dir = os.path.join(HOME_DATA_FOLDER, 'outputs/KD')
elif prediction_mode == 'lobert':
    output_dir = os.path.join(HOME_DATA_FOLDER, 'outputs/LOBERT')
elif prediction_mode == 'benchmark':
    output_dir = os.path.join(PROJECT_FOLDER, f'src/result/glue/benchmark/{sub_dir}')
elif prediction_mode == 'lobert_test':
    output_dir = os.path.join(PROJECT_FOLDER, f'src/result/glue/benchmark/lobert_test')
else:
    raise ValueError('No such prediction mode exists!')

bert_model = os.path.join(HOME_DATA_FOLDER, f'models/pretrained/{bert_model}')
config = BertConfig(os.path.join(bert_model, 'bert_config.json'))
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
args = argparse.Namespace(n_gpu=1,
                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                          fp16=True,
                          eval_batch_size=32,
                          max_seq_length=128)

logger.info(f'sub_dir = {sub_dir}')
logger.info(f'prediction_mode = {prediction_mode}')
logger.info(f'interested_set = {interested_set}')

result_df = pd.read_csv(result_file)

for i, val in result_df.iterrows():
    run_folder = val[0]
    epoch = int(val[1]) - 1
    information = run_folder.split('_')
    task = information[1]
    n_layer = int(information[2].split('.')[1])
    logger.info(f'predicting for task {task}')
    logger.info(f'using model from {run_folder} epoch {epoch}')
    if i < 0:
        logger.info('Skipped!')
        continue

    if interested_task is not None and task not in interested_task:
        logger.info('Skipped because not interested')
        continue

    if 'race' in task:
        args.eval_batch_size = 16
    else:
        args.eval_batch_size = 32

    args.raw_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_raw', 'glue_data', task)
    
    if prediction_mode == 'lobert_test':
        assert lobert_mode != None
        run_folder = os.path.join(LOBERT_DIR, task, lobert_mode, run_folder) # for LoBERT student test benchmark
    else:
        run_folder = os.path.join(KD_DIR, task, sub_dir, run_folder) # for teacher_prediction
    logging.info(f'sub_dir = {sub_dir}')
    encoder_file = glob.glob(run_folder + '/*e.%d.encoder.pkl' % epoch)
    cls_file = glob.glob(run_folder + '/*e.%d.cls.pkl' % epoch)
    assert len(encoder_file) == 1 and len(cls_file) == 1, f'encoder/cls file error: {encoder_file}, {cls_file}'
    encoder_file, cls_file = encoder_file[0], cls_file[0]

    encoder_bert, classifier = init_model(task, output_all_layers, n_layer, config)
    encoder_bert = load_model(encoder_bert, encoder_file, args, 'exact', verbose=True)
    classifier = load_model(classifier, cls_file, args, 'exact', verbose=True)

    all_res = {'train': None, 'dev': None, 'test': None}
    if 'dev' in interested_set or 'valid' in interested_set:
        dev_examples, dev_dataloader, dev_label_ids = get_task_dataloader(task.lower(), 'dev', tokenizer, args,
                                                                          SequentialSampler, args.eval_batch_size)
        dev_res = eval_model_dataloader(encoder_bert, classifier, dev_dataloader, args.device, detailed=True, verbose=False)
        dev_pred_label = dev_res['pred_logit'].argmax(1)
        logger.info('for dev, acc = {}, loss = {}'.format(dev_res['acc'], dev_res['loss']))
        logger.info('debug dev acc = {}'.format((dev_label_ids.numpy() == dev_pred_label).mean()))
        all_res['dev'] = dev_res

    if 'test' in interested_set:
        test_examples, test_dataloader, test_label_ids = get_task_dataloader(task.lower(), 'test', tokenizer, args,
                                                                             SequentialSampler, args.eval_batch_size)
        test_res = eval_model_dataloader(encoder_bert, classifier, test_dataloader, args.device, detailed=True, verbose=False)
        test_pred_label = test_res['pred_logit'].argmax(1)
        logger.info('for test, acc = {}, loss = {}'.format(test_res['acc'], test_res['loss']))
        logger.info('debug test acc = {}'.format((test_label_ids.numpy() == test_pred_label).mean()))

        if task == 'race-merge':
            middle_id = np.array(['middle' in t.mrc_id for t in test_examples])
            logger.info('race-middle test acc = {}'.format((test_label_ids.numpy()[middle_id] == test_pred_label[middle_id]).mean()))
            logger.info('race-hight test acc = {}'.format((test_label_ids.numpy()[~middle_id] == test_pred_label[~middle_id]).mean()))
        all_res['test'] = test_res

    if 'train' in interested_set:
        train_examples, train_dataloader, train_label_ids = get_task_dataloader(task.lower(), 'train', tokenizer, args,
                                                                               SequentialSampler, args.eval_batch_size)
        train_res = eval_model_dataloader(encoder_bert, classifier, train_dataloader, args.device, detailed=True, verbose=False)
        train_pred_label = train_res['pred_logit'].argmax(1)
        logger.info('for training, acc = {}, loss = {}'.format(train_res['acc'], train_res['loss']))
        logger.info('debug train acc = {}'.format((train_label_ids.numpy() == train_pred_label).mean()))
        all_res['train'] = train_res

    if prediction_mode in ['benchmark','lobert_test']:
        if 'race' in task:
            continue

        logger.info('saving benchmark results')
        processor = nli_data_processing.processors[task.lower()]()
        label_list = processor.get_labels()
        test_pred_label = [label_list[tr] for tr in test_res['pred_logit'].argmax(1)]
        test_pred = pd.DataFrame({'index': range(len(test_examples)), 'prediction': test_pred_label})
        if task == 'MNLI':
            test_pred.to_csv(os.path.join(output_dir, task + '-m.tsv'), sep='\t', index=False)
        else:
            test_pred.to_csv(os.path.join(output_dir, task + '.tsv'), sep='\t', index=False)
    elif prediction_mode in ['teacher']:
        logger.info('saving teacher results')
        if not output_all_layers:
            fname = os.path.join(output_dir, task, task + f'_normal_kd_teacher_{n_layer}layer_result_summary.pkl')
        else:
            fname = os.path.join(output_dir, task, task + f'_patient_kd_teacher_{n_layer}layer_result_summary.pkl')
        with open(fname, 'wb') as fp:
            pickle.dump(all_res, fp)
    elif prediction_mode in ['lobert']:
        logger.info('saving lobert_teacher results')
        if not output_all_layers:
            raise ValueError('Lobert teacher should save all output layers!')
        else:
            fname = os.path.join(output_dir, task, task + f'_lobert_teacher_{n_layer}layer_result_summary.pkl')
        with open(fname, 'wb') as fp:
            pickle.dump(all_res, fp)

    logger.info(f'predicting for task {task} Done!')

