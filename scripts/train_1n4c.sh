
cp /root/models/research/object_detection/samples/configs/$MODEL_NAME.config ~
sed -i 's#_BATCH_SIZE_#'`echo $BATCH_SIZE`'#g' ~/$MODEL_NAME.config
sed -i 's#_MODEL_DIR_#'`echo $MODEL_DIR`'#g' ~/$MODEL_NAME.config
sed -i 's#_TRAIN_DATA_DIR_#'`echo $TRAIN_DATA_FILE`'#g' ~/$MODEL_NAME.config
sed -i 's#_LABEL_DIR_#'`echo $LABEL_FILE`'#g' ~/$MODEL_NAME.config
sed -i 's#_EVAL_DATA_DIR_#'`echo $EVAL_DATA_FILE`'#g' ~/$MODEL_NAME.config

PIPELINE_CONFIG_PATH=~/$MODEL_NAME.config
TRAIN_DIR=/output/faster-rcnn/train/

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /root/models/research/object_detection/legacy/train.py \
  --logtostderr \
  --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
  --train_dir=${TRAIN_DIR} \
  --num_clones=4 \
  --ps_tasks=1 | tee /output/faster-rcnn/faster-cnn_1n4c.log