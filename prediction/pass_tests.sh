export OUTPUT_FILE='test_results.txt'
nohup sh -c 'for f in modules/testing/unittests/*.py; do echo "$f" && python -m unittest "$f"; done ' > $OUTPUT_FILE &