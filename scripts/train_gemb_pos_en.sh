path=$1

python2 codes/main.py \
    --train-gemb \
    --config-file models/${path}-config.pkl \
    --dictfile data/$path/dicts.pkl
