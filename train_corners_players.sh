export PYTHONPATH=$PWD:${PWD}/../vball_tracking

# Train on Titan-RTX Turing
export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=1
# export CUDA_VISIBLE_DEVICES=0
# unset WORLD_SIZE

EXP=yolox_x_corners_players
DSET="v9"
TAG=""

CKPT=/mnt/f/output/ByteTrack/YOLOX_outputs/yolox_x_mot_ablation/latest_ckpt.pth.tar
CKPT=/mnt/h/checkpoints/bytetrack/bytetrack_x_mot17.pth.tar
# CKPT=/mnt/h/output/ByteTrack/YOLOX_outputs/yolox_x_fullcourt_v9v9/latest_ckpt.pth.tar
export CUDA_VISIBLE_DEVICES=0,1
EPOCHS=50
python3 tools/train.py \
	-d 2 -b 8 -o --fp16 \
	-f exps/example/mot/${EXP}.py \
	-c $CKPT \
        -expn ${EXP}_${DSET}${TAG}_${EPOCHS}ep \
	train_ann /mnt/g/data/vball/fullcourt/trn_${DSET}_corners.json \
	val_ann /mnt/g/data/vball/fullcourt/val_${DSET}_corners.json \
	max_epoch $EPOCHS no_aug_epochs 4
