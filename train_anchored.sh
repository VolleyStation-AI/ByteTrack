export PYTHONPATH=$PWD:${PWD}/../vball_tracking

# Train on Titan-RTX Turing
export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=0
unset WORLD_SIZE

EXP=yolox_x_anchored
DSET="v3"

CKPT=/mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar
CKPT=/mnt/g/output/ByteTrack/YOLOX_outputs/yolox_x_fullcourt_v8_2/best_ckpt.pth.tar

python3 tools/train.py \
	-d 2 -b 8 --fp16 \
	-f exps/example/mot/${EXP}.py \
	-o -c $CKPT \
        -expn ${EXP}_${DSET}_pt \
	max_epoch 12 no_aug_epochs 4 \
	train_ann /mnt/g/data/vball/anchored_touch/${DSET}_trn.json \
	val_ann /mnt/g/data/vball/anchored_touch/${DSET}_val.json

