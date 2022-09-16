export PYTHONPATH=$PWD:${PWD}/../vball_tracking

# Train on Titan-RTX Turing
export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=0
unset WORLD_SIZE

EXP=yolox_x_tracked_players
DSET=v1
TAG="baseline"

python3 tools/train.py \
	-d 2 -b 8 --fp16 \
	-f exps/example/mot/${EXP}_${DSET}.py \
	-o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar \
        -expn ${EXP}_${DSET}${TAG} \
	max_epoch 12 no_aug_epochs 4


#	train_ann /mnt/g/data/vball/fullcourt/trn_${DSET}.json \
#	val_ann /mnt/g/data/vball/fullcourt/val_${DSET}.json \
