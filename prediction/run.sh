export MODE=$1
export FILE_NUMBER=$2
export DEBUGGING=$3 # allows for debugging by picking a smaller dataset, lower number of epochs, etc
export WRITING_LOGS=$4
export DESCRIPTION=$5

if [ $DEBUGGING = 1 ]
then
    DESCRIPTION="debugging"
fi

export MODULE_NAME=${MODE}_${FILE_NUMBER}
export OUTPUT_FILE=logs/_${MODULE_NAME}_${DESCRIPTION}.out # log file: destination for .out file if nohup is used (example: pneumonia/job_outputs/1.out)

echo "Testing: " $DEBUGGING
echo "File: " $MODULE_NAME
echo "Description: " $DESCRIPTION
echo "Output File: " $OUTPUT_FILE

export DEVICE=-1
if [ $WRITING_LOGS = 1 ]
then
    # nohup
    nohup python -u -m jobs.$MODULE_NAME --debugging "$DEBUGGING"  --description "$DESCRIPTION"  > $OUTPUT_FILE &
else
    #### non-nohup
    CUDA_VISIBLE_DEVICES=$DEVICE python -m jobs.$MODULE_NAME --debugging "$DEBUGGING"  --description "$DESCRIPTION"
fi