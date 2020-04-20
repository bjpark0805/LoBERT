"""
LoBERT: Low rank factorization for BERT

Authors:
 - Bumjoon Park (qkrskaqja@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: src/envs.py
 - Setup the paths for the training.


Version: 1.0

Refer source code from https://github.com/intersun/PKD-for-BERT-Model-Compression.

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

"""

import logging
import os

logger = logging.getLogger(__name__)

PROJECT_FOLDER = os.path.dirname(os.path.dirname(__file__))
HOME_DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
HOME_OUTPUT_FOLDER = os.path.join(HOME_DATA_FOLDER, 'outputs/LMS')
PREDICTION_FOLDER = os.path.join(HOME_DATA_FOLDER, 'outputs/predictions')
