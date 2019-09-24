#!/bin/bash
# The outputs of this script are TFRecord files containing serialized
# tf.Example protocol buffers. See create_coco_tf_record.py for details of how
# the tf.Example protocol buffers are constructed and see
# http://cocodataset.org/#overview for an overview of the dataset.
#
# usage:
#  bash create_coco_tfrecord.sh \

OUTPUT_DIR=/dataset/tfrecord/coco
mkdir -p "${OUTPUT_DIR}"
SCRATCH_DIR=/dataset/mscoco_2017

CURRENT_DIR=/dataset/tf_script/coco

cd ${SCRATCH_DIR}


TRAIN_IMAGE_DIR="${SCRATCH_DIR}/train2017"

VAL_IMAGE_DIR="${SCRATCH_DIR}/val2017"

TEST_IMAGE_DIR="${SCRATCH_DIR}/test2017"

TRAIN_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/instances_train2017.json"

VAL_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/instances_val2017.json"

#TESTDEV_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/image_info_test-dev2017.json"
TESTDEV_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/sample_10_instances_val2017.json"

# Build TFRecords of the image data.
cd "${CURRENT_DIR}"
python create_coco_tf_record.py \
  --logtostderr \
  --include_masks \
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --test_image_dir="${TEST_IMAGE_DIR}" \
  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
  --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}"

