export PYTHONPATH=$PWD:${PWD}/../vball_tracking

# Train on Titan-RTX Turing
export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=1
# export CUDA_VISIBLE_DEVICES=0
# unset WORLD_SIZE

EXP=yolox_x_fullcourt
DSET="v9"
TAG=$DSET

CKPT=/mnt/f/output/ByteTrack/YOLOX_outputs/yolox_x_mot_ablation/latest_ckpt.pth.tar
CKPT=/mnt/h/checkpoints/bytetrack/bytetrack_x_mot17.pth.tar
export CUDA_VISIBLE_DEVICES=0,1
#	-d 2 -b 8 --fp16 \
python3 tools/train.py \
	-d 2 -b 8 -o --fp16 \
	-f exps/example/mot/${EXP}.py \
	-c $CKPT \
        -expn ${EXP}_${DSET}${TAG}_testspeed \
	train_ann /mnt/g/data/vball/fullcourt/trn_${DSET}.json \
	val_ann /mnt/g/data/vball/fullcourt/val_${DSET}.json \
	max_epoch 12 no_aug_epochs 4

#	-o \
