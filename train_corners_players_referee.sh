export PYTHONPATH=$PWD:${PWD}/../vball_tracking
export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=1
# export CUDA_VISIBLE_DEVICES=0
# unset WORLD_SIZE

EXP=yolox_x_corners_players_referee
DSET="v10"
TAG=""

CKPT=/mnt/f/output/ByteTrack/YOLOX_outputs/yolox_x_mot_ablation/latest_ckpt.pth.tar
CKPT=bytetrack_x_mot17.pth.tar
CKPT=/home/pokropow/volleystation/output/ByteTrack/YOLOX_outputs/yolox_x_corners_players_referee_v9_50ep/latest_ckpt.pth.tar
export CUDA_VISIBLE_DEVICES=0
EPOCHS=50
# python tools/train.py \
# 	-d 1 -b 4 -o --fp16 \
# 	-f exps/example/mot/${EXP}.py \
# 	--resume \
# 	-c $CKPT \
#         -expn ${EXP}_${DSET}${TAG}_${EPOCHS}ep \
# 	train_ann /home/pokropow/volleystation/datasets/player_detection/fullcourt/trn_${DSET}_corners_referee.json \
# 	val_ann /home/pokropow/volleystation/datasets/player_detection/fullcourt/val_${DSET}_corners_referee.json \
# 	max_epoch $EPOCHS no_aug_epochs 4
# --resume \
python tools/train.py \
	-d 1 -b 4 -o --fp16 \
	-f exps/example/mot/${EXP}.py \
	-c $CKPT \
	--resume \
    -expn ${EXP}_${DSET}${TAG}_${EPOCHS}ep_test_test \
	train_ann /home/pokropow/volleystation/datasets/player_detection/trn_v10.json \
	val_ann /home/pokropow/volleystation/datasets/player_detection/val_v10.json \
	max_epoch $EPOCHS no_aug_epochs 4