export PYTHONPATH=$PWD:${PWD}/../vball_tracking

# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt.py -b 2 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/yolox_x.pth

# higher lr
# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v2.py -b 2 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar -expn yolox_x_fullcourt_v2


# longer 
# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v2_24ep_b.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar -expn yolox_x_fullcourt_v2_24ep_b

# longer with mot17 as pretrained
# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v4_24ep.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar -expn yolox_x_fullcourt_v4_nov_24ep

# longer with imgnet as pretrained
# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v2_24ep.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/yolox_x.pth -expn yolox_x_imgnet

# from scratch
# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v3_24ep.py -b 4 --fp16 -o  -expn yolox_x_scratch


#python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v4_24ep.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar \
#	-expn yolox_x_fullcourt_v7_0 \
#	train_ann /mnt/g/data/vball/fullcourt/trn_v7_0.json \
#	val_ann /mnt/g/data/vball/fullcourt/val_v7_0.json \
#	max_epoch 24

python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v4_24ep.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar \
	-expn yolox_x_fullcourt_v7_2 \
	train_ann /mnt/g/data/vball/fullcourt/trn_v7_2.json \
	val_ann /mnt/g/data/vball/fullcourt/val_v7_2.json \
	max_epoch 12 no_aug_epochs 2

python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v4_24ep.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar \
	-expn yolox_x_fullcourt_v7_3 \
	train_ann /mnt/g/data/vball/fullcourt/trn_v7_3.json \
	val_ann /mnt/g/data/vball/fullcourt/val_v7_3.json \
	max_epoch 12 no_aug_epochs 2

# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_adam.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar \

#python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_adam.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/yolox_x.pth \
#	-expn yolox_coco_fullcourt_adam_v7_1 \
#	train_ann /mnt/g/data/vball/fullcourt/trn_v7_1.json \
#	val_ann /mnt/g/data/vball/fullcourt/val_v7_1.json \
#	max_epoch 12 no_aug_epochs 4

#python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_adam.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/yolox_x.pth \
#	-expn yolox_coco_fullcourt_adam_v7_2 \
#	train_ann /mnt/g/data/vball/fullcourt/trn_v7_2.json \
#	val_ann /mnt/g/data/vball/fullcourt/val_v7_2.json \
#	max_epoch 12 no_aug_epochs 2


#python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_adam.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/yolox_x.pth \
#	-expn yolox_coco_fullcourt_adam_v7_3 \
#	train_ann /mnt/g/data/vball/fullcourt/trn_v7_3.json \
#	val_ann /mnt/g/data/vball/fullcourt/val_v7_3.json \
#	max_epoch 12 no_aug_epochs 4

#python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_adam.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/yolox_x.pth \
#	-expn yolox_coco_fullcourt_adam_v7_3_24 \
#	train_ann /mnt/g/data/vball/fullcourt/trn_v7_3.json \
#	val_ann /mnt/g/data/vball/fullcourt/val_v7_3.json \
#	max_epoch 24

#python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v4_24ep.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar \
#	-expn yolox_x_fullcourt_v7_2 \
#	train_ann /mnt/g/data/vball/fullcourt/trn_v7_2.json \
#	val_ann /mnt/g/data/vball/fullcourt/val_v7_2.json \
#	max_epoch 12 no_aug_epochs 4

