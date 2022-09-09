### - Set up -

Versions

Python: 3.7
Tensorflow: 3.7
Keras: 2.7
Tensorboard: 2.9.1


To-do's:
- change cache and data root paths in modules/main/parameters.py
- install libraries, can be found in requirements.txt (generated with pipreqs)


### - File Execution Scipts -

offers options to execute a file in the jobs folder, launch debugging mode, write to an output file or standard output, and adds a description used for logging and in file.<br/>
- debugging mode executes job with less data, lower number of epochs, etc, to ensure every step runs.

*bash run.sh $MODE $FILE_NUMBER #DEBUGGING $WRITING_LOGS #DESCRIPTION*<br/>
:ex: bash run.sh predict 6 1 0 "2.3.23" to run predict_6 in debugging model without nohup<br/>
:ex: bash run.sh train 80 0 1 "_varing_lr" to run train_80 with nohup with description "_varing_lr'

### - Other Scripts -

analyze the metrics of a job in tensorboard by passing correct job folder.

*bash view.sh $LOG_DIR*<br/>
:ex: bash view.sh /home/alirachidi/doordash/prediction/cache/train_81/1


### - Caching and logging -

a cache folder is created and name after each train/predict file (with proxy folders created in debug mode. <br/>
a file (!= job) is susceptible of running multiple jobs (e.g., kfold), therefore each file has a different
subfolder for each job, numbered 0 to N based on order. Functions initialize_file_folder() and initialize_job() play key role in maintaing the structure.<br/>
The entire standard output can be routed via nohup and accessed in the logs folders, where details on training are constantly reported throughout the execution of the file.

### - Data -

To increase run speed, we generate features based on a set of parameters, once at a time, and write them to a file for later convenient use.<br/>
Each file corresponds to an experiment. We can choose to generate as many features for as many samples as desired. They are code by version number i.e. v1, v2, etc, with the most relevant being v4.<br/>
Feature files are read in the order in which they were written, since feature generation is done with data ordered by domain, while writing and reading is done separately by domain, leading to 2 files, market and store features files, for each version.<br/>
back up files are generated, coded with the date of creation.
see feature_engineering.py for more

### - Modules Library -

I wrote modules, a library containg 9 utility files, a few templates for neural networks, and the testing
submodule below.<br/>
the  parameters utility is a global tool to track key parameters across the jobs and files, some of which
may be fixed (i.e., file description) while others need to be constantly updated (i.e., job id indicating
job folder, varying training parameters).<br/>
the helpers utility is the tool for the main job file It contains many functions involving file and job management, kfold options, job functions and custom training functions, and more.<br/>
It works in great tandem with the parameters utility.

### - Testing -

the testing script will run every script in the testing/unittests folders and output the results to the log file, test_results.txt.<br/>
I primarly test my feature engineering, stacking and search space manipulations tools. my process is to write custom orders with correct set of generated features in a dataframe, that I use as an input to my test functions, by for example regenerating features and comparing them against truth value.<br/>
Tests are named X_Y, with X a digit indicative of the example/datafrane used, while Y refers to the different tests applied to that example.<br/>
TODO: fix test 1_2 and 1_3, still working custom test cases, but require update after code change.

*bash pass_test.sh*

### - Data Analysis and Results -

I conducted a thorough data analysis in the data_analysis notebook under jobs.<br/>
It contains all types of plots revealing exciting insights.<br/>
One can also find a summary of validation and test results at the bottom of the notebook along
with h5 model locations, my drift analysis and more.