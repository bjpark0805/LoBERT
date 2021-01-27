# LoBERT: Low-rank factorization of BERT

This package provides an implementation of LoBERT, which is to compress BERT
through various techniques for low-rank factorizations of matrices and tensors.

## Overview

This repository is based on the [GitHub repository](https://github.com/intersun/PKD-for-BERT-Model-Compression)
for [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355).
All source files are from the repository if not mentioned otherwise.
The main scripts that actually run tasks are the following two files, and they 
have been modified from the original files in the original repository:
- `NLI_KD_training.py`
- `run_glue_benchmark.py` 

The following four files have been newly created.
The first three files in the `script` directory automatically runs the main
script above for specific objectives such as to train teachers or students.
The last file `load_student.py` is the core module of this project which implements various
functions for compressing BERT via low-rank factorizations, including Tucker
decomposition and SVD.
The only public function in the file, `lobert_student()`,  is called by
`NLI_KD_training.py` to initialize compressed models.
All other functions are called only in `load_student.py`.      
- `scripts/finetune_teacher.py`
- `scripts/save_teacher_prediction.py`
- `scripts/train_student.py`
- `utils/load_student.py`

The tree structure of this project is given as follows:

``` Unicode
LoBERT
  ├── data
  │    ├── data_raw
  │    │     ├── glue_data: task dataset
  │    │     └── download_glue_data.py
  │    ├── models
  │    │     └── bert_base_uncased: ckpt
  │    └── outputs
  │          ├── LOBERT: save teacher model prediction & trained student model 
  │          └── KD: save fine-tuned BERT-base
  ├── src
  │    ├── bert
  │    │     ├── examples: BERT utils 
  │    │     ├── pytorch_pretrained_bert: BERT sturcture files
  │    │     └── tests: BERT test files
  │    ├── result
  │    │     └── glue/result_summary: save result_summaries for each hyperparameter
  │    ├── scripts
  │    │     ├── finetune_teacher.py : finetune teacher model for each task
  │    │     ├── save_teacher_prediction.py: save teacher prediction for all the layers 
  │    │     └── train_student.py: train student model 
  │    ├── utils
  │    │     └── load_student.py: create & initiate student model
  │    ├── envs.py: save directory paths for several usage
  │    ├── NLI_KD_training.py: comprehensive training file for teacher and student models
  └──  └── run_glue_benchmark.py: comprehensive prediction file for teacher and student models
```

#### Data description
- GLUE datasets

* Note that: 
    * GLUE datasets consists of CoLA, diagnostic, MNLI, MRPC, QNLI, QQP, RTE, SNLI, SST-2, STS-B, WNLI
    * You can download GLUE datasets by LoBERT/data/data_raw/download_glue_data.py
   
#### Output
* The trained model will be saved in `data/outputs/LOBERT/{task}` after training.

## Install

#### Environment 
* Ubuntu
* CUDA 10.0
* Python 3.6
* numpy
* torch
* Tensorly
* tqdm
* pandas
* apex

## How to Run

### Clone the repository

```
git clone https://github.com/bjpark0805/LoBERT.git
cd LoBERT
```

### Demo

You can easily test the whole procedure by running `demo.sh`.
The script will fine-tune BERT-base for MRPC, save its predictions, and train a
student model which is compressed by Truncated SVD.
This is run only on machines with GPUs, because of an assert statement in
`finetune_teacher.py`, which is run first in `demo.sh`. 

In case the following error, this [link](https://github.com/NVIDIA/apex/issues/116) will help you to solve it.
```
TypeError: Class advice impossible in Python3. Use the @Implementer class decorator instead.
```

### Training & Testing

* To fine-tune the teacher model(BERT-Base) on `data/output/LOBERT/{task}` run script:
    ```    
    cd src/scripts
    python finetune_teacher.py
    ```
    The trained model will be saved in `data/output/LOBERT/{task}`
* To save the teacher model's predictions, run script:
    ```
    cd src/scripts
    python save_teacher_prediction.py
    ```
    The teacher prediction will be saved in `data/output/LOBERT/{task}/{task}_lobert_teacher_12layer_result_summary.pkl`.
* To train the student model, run script:
    ```
    cd src/scripts
    python train_student {task} {student_mode}
    ```
    * For example:
        * python train_student MRPC self_svd : decompose with Truncated SVD on Query, Key, Value each.
        * python train_student MRPC self_tuckerlayer : decompose with Tucker on stacked Query, Key, Value for each layer
        * python train_student MRPC self_falconlayer : decompose with FALCON on stacked Query, Key, Value for each layer
        * python train_student MRPC self_tuckervertical : decompose with Tucker on stacked Query for all the layers (same for Key, Value)
        * python train_student MRPC self_falconvertical : decompose with FALCON on stacked Query for all the layers (same for Key, Value)
        * python train_student MRPC self_tucker_4d : decompose with Tucker on stacked Query,Key,Value stacked for all layers
        * python train_student MRPC self_falcon_4d : decompose with FALCON on stacked Query,Key,Value stacked for all layers
        * python train_student MRPC ffn_svd : decompose with Truncated SVD on Intermediate, Output each
        * python train_student MRPC ffn_tucker : decompose with Tucker on stacked Intermediate for all the layers(same for Output) 


### Attention
There are three python files in the demo.sh.
Please check if the directories are correct

The output of finetune_teacher.py is in the ./data/outputs/kd
The output of save_teacher_prediction.py is in the ./data/outputs/LOBERT


## Contact

- Bumjoon Park (qkrskaqja@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

*This software may be used only for research evaluation purposes.*  
*For other purposes (e.g., commercial), please contact the authors.*
