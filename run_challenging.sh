
CHALLENGING_VIDS=(20211124_usc_stanford_0.mp4 20211014_tcu_texas_0.mp4 20210919_kentucky_stanford_1.mp4 20210919_kentucky_stanford_0.mp4 20211001_arizonastate_stanford_1.mp4 20211001_arizonastate_stanford_0.mp4 20211002_olemiss_florida_1.mp4 20211002_olemiss_florida_0.mp4)

for VID in "${CHALLENGING_VIDS[@]}"
do
    VID_PATH=/home/atao/devel/volleyvision/challenging_vids/${VID}
    python tools/demo_track.py video --path $VID_PATH -f exps/example/mot/yolox_x_mix_det.py -c /mnt/g/checkpoints/ByteTrack/bytetrack_x_mot17.pth.tar   --fp16 --fuse --save_result --tag challenging

done
