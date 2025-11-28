export PYTHONPATH=$PWD:${PWD}/../multi/vball_tracking

#export CUDA_VISIBLE_DEVICES=2,3
#export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
unset WORLD_SIZE

EXP=yolox_x_corners_players_referee
DSET="v11_player_ref_court"
TAG="_mc_eval-4090"

CKPT=/mnt/f/output/ByteTrack/YOLOX_outputs/yolox_x_mot_ablation/latest_ckpt.pth.tar
CKPT=/mnt/h/checkpoints/bytetrack/bytetrack_x_mot17.pth.tar
python3 tools/train.py \
	-d 1 -b 4 -o --fp16 \
	-f exps/example/mot/${EXP}.py \
	-c $CKPT \
        -expn ${EXP}_${DSET}${TAG} \
	train_ann /mnt/g/data/vball/fullcourt/val_${DSET}.json \
	val_ann /mnt/g/data/vball/fullcourt/val_${DSET}.json \
	max_epoch 50 no_aug_epochs 8 \
	data_num_workers 8
