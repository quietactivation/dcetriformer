MODEL='model_name'
# DATASET_TYPE='ProstateX_dce_h5'
# DATASET_TYPE='Oscar_345_dce_2_3_bkp'
DATASET_TYPE='ProstateX_dce_118_copy'
# DATASET_TYPE='Oscar_345_PD_h5'
# DATASET_TYPE='PROSTATE_MRI_160_h5'
# DATASET_TYPE='ProstateX_h5_dce15'
BASE_PATH='/media/Data/MRI'
# BASE_PATH='/media/data16TB'

RECONS_KEY='DCE_02'
# RECONS_KEY='DCE2'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}
# TARGET_PATH=${BASE_PATH}'/datasets/PROSTATE_MRI_data/'${DATASET_TYPE}
# TARGET_PATH=${BASE_PATH}'/ProstateX_160X160/'${DATASET_TYPE}
PREDICTIONS_PATH='pred_path/'${MODEL}'/results_dce_2'
REPORT_PATH='report_path/'${MODEL}'/report_'${RECONS_KEY}'.txt'

echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --recons-key ${RECONS_KEY}


RECONS_KEY='DCE_03'
# RECONS_KEY='DCE3'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}
# TARGET_PATH=${BASE_PATH}'/datasets/PROSTATE_MRI_data/'${DATASET_TYPE}
# TARGET_PATH=${BASE_PATH}'/ProstateX_160X160/'${DATASET_TYPE}
PREDICTIONS_PATH='pred_path/'${MODEL}'/results_dce_3'
REPORT_PATH='report_path/'${MODEL}'/report_'${RECONS_KEY}'.txt'

echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --recons-key ${RECONS_KEY}

