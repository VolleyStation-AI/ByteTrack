export PYTHONPATH=$PWD:${PWD}/../vball_tracking
export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=1
# export CUDA_VISIBLE_DEVICES=0
# unset WORLD_SIZE

EXP=yolox_bev
DSET="v0"

CKPT=/mnt/h/output/best_models/bytetrack/yolox_x_corners_players_v9_50ep/latest_ckpt.pth.tar
export CUDA_VISIBLE_DEVICES=0,1
EPOCHS=50
python3 tools/train.py \
	-d 1 -b 4 -o --fp16 \
	-f exps/example/mot/${EXP}.py \
	-c $CKPT \
        -expn ${EXP}_${DSET}_${EPOCHS}ep \
	train_ann /mnt/g/data/vball/fullcourt/bev/trn_bev_${DSET}.json \
	val_ann /mnt/g/data/vball/fullcourt/bev/val_bev_${DSET}.json \
	max_epoch $EPOCHS no_aug_epochs 4
