export PYTHONPATH=$PWD:${PWD}/../vball_tracking
export CUDA_VISIBLE_DEVICES=2,3
export WORLD_SIZE=1
# export CUDA_VISIBLE_DEVICES=0
# unset WORLD_SIZE

EXP=yolox_x_corners_players
DSET="v9"
TAG=""

CKPT=/mnt/f/output/ByteTrack/YOLOX_outputs/yolox_x_mot_ablation/latest_ckpt.pth.tar
CKPT=/mnt/h/checkpoints/bytetrack/bytetrack_x_mot17.pth.tar
# CKPT=/mnt/h/output/ByteTrack/YOLOX_outputs/yolox_x_fullcourt_v9v9/latest_ckpt.pth.tar
export CUDA_VISIBLE_DEVICES=2,3
EPOCHS=50

ROOT=/mnt/g/data/vball/fullcourt
ROOT=/mnt/t/data/vball/fullcourt
TRN_JSON=${ROOT}/trn_${DSET}_corners.json
VAL_JSON=${ROOT}/val_${DSET}_corners.json

python3 tools/train.py \
	-d 2 -b 8 -o --fp16 \
	-f exps/example/mot/${EXP}.py \
	-c $CKPT \
        -expn ${EXP}_${DSET}${TAG}_${EPOCHS}ep \
	root $ROOT train_ann $TRN_JSON val_ann $VAL_JSON \
	max_epoch $EPOCHS no_aug_epochs 4
