MODEL='DCEtriformer_rectALT_2tp'
# DATASET_TYPE='ProstateX_dce_h5'
# DATASET_TYPE='Oscar_345_dce_2_3_bkp'
DATASET_TYPE='ProstateX_dce_118_copy'
# DATASET_TYPE='Oscar_345_PD_h5'
# DATASET_TYPE='PROSTATE_MRI_160_h5'
# DATASET_TYPE='ProstateX_h5_dce15'
BASE_PATH='/media/Data/MRI'
# BASE_PATH='/media/data16TB'

CHECKPOINT='/dir/'${MODEL}'/model.pt'
OUT_DIR_DCE_2='/dir/'${MODEL}'/results_dce_2'
OUT_DIR_DCE_3='/dir/'${MODEL}'/results_dce_3'

# VALIDATION_PATH=${BASE_PATH}'/datasets/PROSTATE_MRI_data/'${DATASET_TYPE}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}
# VALIDATION_PATH=${BASE_PATH}'/ProstateX_160X160/'${DATASET_TYPE}
BATCH_SIZE=1
DEVICE='cuda:0'

# echo python valid.py --checkpoint ${CHECKPOINT}  --out_dir-ktrans ${OUT_DIR_KTRANS} --batch-size ${BATCH_SIZE} --device ${DEVICE} --validation-path ${VALIDATION_PATH} 
# python valid.py --checkpoint ${CHECKPOINT} --out_dir-ktrans ${OUT_DIR_KTRANS} --batch-size ${BATCH_SIZE} --device ${DEVICE} --validation-path ${VALIDATION_PATH}  

echo python valid.py --checkpoint ${CHECKPOINT}  --batch-size ${BATCH_SIZE} --device ${DEVICE} --validation-path ${VALIDATION_PATH} --out-dir-dce-2 ${OUT_DIR_DCE_2} --out-dir-dce-3 ${OUT_DIR_DCE_3} #--out-dir-dce-15 ${OUT_DIR_DCE_15}
python valid.py --checkpoint ${CHECKPOINT}  --batch-size ${BATCH_SIZE} --device ${DEVICE} --validation-path ${VALIDATION_PATH} --out-dir-dce-2 ${OUT_DIR_DCE_2} --out-dir-dce-3 ${OUT_DIR_DCE_3} #--out-dir-dce-15 ${OUT_DIR_DCE_15}

