export PYTHONPATH=./:$PYTHONPATH
model_dir=./models/test11_cifar100/
CUDA_VISIBLE_DEVICES=5 python3 fs_eval_cifar100.py \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=pgd-cw \
    --dataset=cifar100 \
    --batch_size_test=120 \
    --resume
