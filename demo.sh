# This script is to run a demo of LoBERT. This fine-tune BERT-base for MRPC,
# save the parent model's prediction, and train a student model by Truncated SVD.
# Compression rate is 50% by default.

python src/scripts/finetune_teacher.py
python src/scripts/save_teacher_prediction.py
python src/scripts/train_student.py MRPC self_svd
