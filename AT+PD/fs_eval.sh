export PYTHONPATH=./:$PYTHONPATH
model_dir=./models/test1/
CUDA_VISIBLE_DEVICES=5 python3 fs_eval.py \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=pgd-cw \
    --dataset=cifar10 \
    --batch_size_test=120 \
    --resume
