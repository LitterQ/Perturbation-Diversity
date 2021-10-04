export PYTHONPATH=./:$PYTHONPATH
model_dir=./models/test5_cifar100/
mkdir -p $model_dir
CUDA_VISIBLE_DEVICES=4 python fs_main_cifar100.py \
    --resume \
    --adv_mode='feature_scatter' \
    --lr=0.1 \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --max_epoch=600 \
    --save_epochs=1 \
    --decay_epoch1=60 \
    --decay_epoch2=90 \
    --batch_size_train=120 \
    --dataset=cifar100

