export PYTHONPATH=./:$PYTHONPATH
model_dir=./models/test7/
CUDA_VISIBLE_DEVICES=4 python3 fs_eval.py \
    --model_dir=$model_dir \
    --init_model_pass=60 \
    --attack=True \
    --attack_method_list=pgd-cw \
    --dataset=cifar10 \
    --batch_size_test=120 \
    --resume
