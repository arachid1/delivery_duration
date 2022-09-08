if [ $# -eq 0 ]
then
    echo "No arguments supplied"
    export LOG_DIR=../../neuralink/classification/cache/train_3/1
    tensorboard --logdir=$LOG_DIR --reload_interval=5
else
    export LOG_DIR=$1
    tensorboard --logdir=$LOG_DIR --reload_interval=5
fi

echo "Log Directory: $LOG_DIR"
