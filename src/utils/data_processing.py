"""
LoBERT: Low rank factorization for BERT

Authors:
 - Bumjoon Park (qkrskaqja@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: src/utils/data_processing.py
 - Process task data.


Version: 1.0

Refer source code from https://github.com/intersun/PKD-for-BERT-Model-Compression.

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

"""


from utils.nli_data_processing import init_glue_model, get_glue_task_dataloader
from utils.race_data_processing import init_race_model, get_race_task_dataloader


def init_model(task_name, output_all_layers, num_hidden_layers, config):
    if 'race' in task_name.lower():
        return init_race_model(task_name, output_all_layers, num_hidden_layers, config)
    else:
        return init_glue_model(task_name, output_all_layers, num_hidden_layers, config)


def get_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size=None, knowledge=None, extra_knowledge=None):
    if 'race' in task_name.lower():
        return get_race_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge)
    else:
        return get_glue_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge)

