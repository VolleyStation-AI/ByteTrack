DSET="clean_v10_marek_player_ref_court"
DSET="clean_v10_player_ref_court"
DSET="marek_v10"

OUTROOT=/mnt/h/output/trn/ByteTrack/YOLOX_outputs
CKPT=${OUTROOT}/yolox_x_corners_players_referee_clean_v10_player_ref_court/latest_ckpt.pth.tar
CKPT=${OUTROOT}/yolox_x_corners_players_referee_v10_50ep/latest_ckpt.pth.tar
CKPT=${OUTROOT}/yolox_x_corners_players_referee_clean_v10_player_ref_court_50ep/last_epoch_ckpt.pth.tar

TRN_ANN=/mnt/g/data/vball/fullcourt/trn_${DSET}.json
VAL_ANN=/mnt/g/data/vball/fullcourt/val_${DSET}.json

python eval_per_class.py \
       --ckpt $CKPT \
       -f /home/atao/vsdevel/multi/ByteTrack/exps/example/mot/yolox_x_corners_players_referee.py \
       --fp16 -expn test_multicls_eval \
       -b 4 \
       train_ann $VAL_ANN \
       val_ann $VAL_ANN 


# python eval_per_class.py -b 4 --ckpt /mnt/h/output/trn/ByteTrack/YOLOX_outputs/yolox_x_corners_players_referee_clean_v10_player_ref_court_50ep/last_epoch_ckpt.pth.tar -f /home/atao/vsdevel/multi/ByteTrack/exps/example/mot/yolox_x_corners_players_referee.py --fp16  -expn test_multicls_eval  train_ann /mnt/g/data/vball/fullcourt/trn_${DSET}.json val_ann /mnt/g/data/vball/fullcourt/val_${DSET}.json 
