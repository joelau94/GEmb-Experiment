path=$1

python2 codes/preprocess.py \
    --dictfile data/$path/dicts.pkl \
    --train-data-file data/$path/train.pkl \
    --dev-data-file data/$path/dev.pkl \
    --test-data-file data/$path/test.pkl \
    --raw-train-file data/$path/train.txt \
    --raw-dev-file data/$path/dev.txt \
    --raw-test-file data/$path/test.txt \
    --task tagging \
    --min-freq 2
