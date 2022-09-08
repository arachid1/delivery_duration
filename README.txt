- Set up -
pip requirements.txt
conda
version

- File Execution Scipts -

options to execute a file in the jobs folder, in debugging mode or not, writing to a file or standard output,
and with a description
debugging mode executes job with less data, lower number of epochs, etc, to ensure every step runs
bash run.sh $MODE $FILE_NUMBER #DEBUGGING $WRITING_LOGS #DESCRIPTION
:ex: bash run.sh predict 6 1 0 "2.3.23" to run predict_6 in debugging model without nohup
:ex: bash run.sh train 80 0 1 "_varing_lr" to run train_80 with nohup with description "_varing_lr'

- Other Scripts -

analyze the metrics of a job in tensorboard by passing correct cache job folder
bash view.sh $LOG_DIR
:ex: bash view.sh /home/alirachidi/doordash/prediction/cache/train_81/1

- Caching, job directories and logs -

a cache folder is created and name after each train/predict file (with separate folders in debug mode)
a file (!= job) is susceptible of running multiple jobs (e.g., kfold), therefore each file has a different
subfolder for each job, numbered 0 to 5 based on order.
The entire standard output can be routed via nohup and seen in the logs folders, where details on training
are  constantly reported throughout the execution of the file.

- Data -

to increase run speed, we generate features based on a set scheme once and write their files to a text file
in the data folder, later read at convenience in training codes.
they are code by version number i.e. v1, v2, v3, ...
the code is written such that the files must be read in the order in which they were written
debugging files and back up files are generated as well
see feature_engineering.py for more

- Modules Library -

I wrote modules, a library containg 9 utility files, a few templates for neural networks, and the testing
submodule below.
the  parameters utility is a global tool to track key parameters across the jobs and filers, some of which
may be fixed (i.e., file description) while others need to be constantly updated (i.e., job id indicating
job folder)
the helpers utility is the tool for the main job file, train or predict. It contains many functions from
file and job management, kfold options, job functions and custom training functions, and more. It works increase
tandem with the parameters utility

- Testing -

the testing script will run every test script in the testing/unittests folders and output its logs to
test_results.txt
I primarly test my feature engineering, stacking and search space manipulations tools
has its own caching folder
TODO: fix test 1_2 and 1_3, still working custom test cases, but require update after code change
bash pass_test.sh

- Data Analysis and Results -

I conducted a thorough data analysis in the data_analysis notebook under jobs. it has its own cache folder.
It contains all types of plots revealing exciting insights.
One can also find a summary of validation and test results at the bottom of the notebook along
with h5 model locations, my drift analysis and more.
