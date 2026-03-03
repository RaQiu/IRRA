#!/bin/bash
DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name iira \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \

# #!/bin/bash
# DATASET_NAME="CUHK-PEDES"

# # GPU 0 -> None 模式（基准实验）
# CUDA_VISIBLE_DEVICES=0 MY_RANK=0 python train.py \
# --name irra_none --img_aug --batch_size 64 --MLM --dataset_name $DATASET_NAME \
# --loss_names 'sdm+mlm+id' --num_epoch 30 &

# # GPU 1 -> Image Noise 模式
# CUDA_VISIBLE_DEVICES=1 MY_RANK=1 python train.py \
# --name irra_img --img_aug --batch_size 64 --MLM --dataset_name $DATASET_NAME \
# --loss_names 'sdm+mlm+id' --num_epoch 30 &

# # GPU 2 -> Text Noise 模式
# CUDA_VISIBLE_DEVICES=2 MY_RANK=2 python train.py \
# --name irra_text --img_aug --batch_size 64 --MLM --dataset_name $DATASET_NAME \
# --loss_names 'sdm+mlm+id' --num_epoch 30 &

# wait
# echo "✅ 三个实验已独立运行完成，数据保存在各自的 logs 目录下。"