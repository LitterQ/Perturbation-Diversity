export PYTHONPATH=./:$PYTHONPATH
model_dir=./models/test3_svhn/
CUDA_VISIBLE_DEVICES=3 python3 fs_eval_svhn.py \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=pgd-cw \
    --dataset=svhn \
    --batch_size_test=120 \
    --resume
