MODEL='model_name'
# DATASET_TYPE='ProstateX_dce_h5'
DATASET_TYPE='ProstateX_dce_118_copy'
# DATASET_TYPE='ProstateX_h5_dce15'
# DATASET_TYPE='Oscar_345_PD_h5'
# DATASET_TYPE='Oscar_345_dce_2_3_bkp'
BASE_PATH='/media/Data/MRI'
# BASE_PATH='/media/data16TB'
BATCH_SIZE=4
NUM_EPOCHS=200
DEVICE='cuda:0'
EXP_DIR='exp_dir/'${MODEL}

TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}


echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH}  #--resume --checkpoint ${CHECKPOINT}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH}  #--resume --checkpoint ${CHECKPOINT}
# --resume --checkpoint ${CHECKPOINT} /media/data16TB/ProstateX_160X160/Oscar_345_PD_h5