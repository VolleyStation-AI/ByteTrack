export PYTHONPATH=$PWD:${PWD}/../vball_tracking

export CUDA_VISIBLE_DEVICES=0,1
unset WORLD_SIZE

EXP=yolox_x_fullcourt
TAG=dancetrack
DSET=v8_2

python3 tools/train.py -d 2 -f exps/example/mot/${EXP}_${DSET}.py -b 8 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/dancetrack.pth.tar \
        -expn ${EXP}_${DSET}_${TAG} \
	train_ann /mnt/g/data/vball/fullcourt/trn_${DSET}.json \
	val_ann /mnt/g/data/vball/fullcourt/val_${DSET}.json \
	max_epoch 12 no_aug_epochs 2

