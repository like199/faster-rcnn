
cp /root/models/research/object_detection/samples/configs/$MODEL_NAME.config ~
sed -i 's#_BATCH_SIZE_#'`echo $BATCH_SIZE`'#g' ~/$MODEL_NAME.config
sed -i 's#_MODEL_DIR_#'`echo $MODEL_DIR`'#g' ~/$MODEL_NAME.config
sed -i 's#_TRAIN_DATA_DIR_#'`echo $TRAIN_DATA_FILE`'#g' ~/$MODEL_NAME.config
sed -i 's#_LABEL_DIR_#'`echo $LABEL_FILE`'#g' ~/$MODEL_NAME.config
sed -i 's#_EVAL_DATA_DIR_#'`echo $EVAL_DATA_FILE`'#g' ~/$MODEL_NAME.config

PIPELINE_CONFIG_PATH=~/$MODEL_NAME.config
TRAIN_DIR=/output/faster-rcnn/train/
EVAL_DIR=~/output/faster-rcnn/eval_dir
python3 ~/models/research/object_detection/legacy/eval.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --eval_dir=${EVAL_DIR} \
    --logtostderr \
    --checkpoint_dir=${TRAIN_DIR}