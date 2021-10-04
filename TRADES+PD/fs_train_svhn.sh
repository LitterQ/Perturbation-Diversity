export PYTHONPATH=./:$PYTHONPATH
#model_dir=./models/trades_baseline_svhn_2/
model_dir=./models/test2_svhn/
mkdir -p $model_dir
CUDA_VISIBLE_DEVICES=3 python fs_main_svhn.py \
    --resume \
    --adv_mode='feature_scatter' \
    --lr=0.1 \
    --model_dir=$model_dir \
    --init_model_pass=60 \
    --max_epoch=600 \
    --save_epochs=1 \
    --decay_epoch1=60 \
    --decay_epoch2=90 \
    --batch_size_train=120 \
    --dataset=svhn

