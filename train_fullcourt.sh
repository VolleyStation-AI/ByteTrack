export PYTHONPATH=${PWD}:${PWD}/../vball_tracking

# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt.py -b 2 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/yolox_x.pth

# higher lr
# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v2.py -b 2 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar -expn yolox_x_fullcourt_v2


# longer 
# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v2_24ep_b.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar -expn yolox_x_fullcourt_v2_24ep_b

# longer with mot17 as pretrained
python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v4_24ep.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar -expn yolox_x_fullcourt_v4_24ep

# longer with imgnet as pretrained
# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v2_24ep.py -b 4 --fp16 -o -c /mnt/g/checkpoints/ByteTrack/yolox_x.pth -expn yolox_x_imgnet

# from scratch
# python3 tools/train.py -f exps/example/mot/yolox_x_fullcourt_v3_24ep.py -b 4 --fp16 -o  -expn yolox_x_scratch
