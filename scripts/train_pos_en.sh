path=$1

python2 codes/main.py \
    --config-file models/${path}-config.pkl \
    --use-gemb \
    --train-data-file data/$path/train.pkl \
    --dev-data-file data/$path/dev.pkl \
    --test-data-file data/$path/test.pkl \
    --task tagging \
    --keep-prob 0.9 \
    --dictfile data/$path/dicts.pkl \
    --embed-dim 300 \
    --hidden-dims 256 256 \
    --ckpt models/${path}-model \
    --max-ckpts 20 \
    --batch-size 32 \
    --max-steps 100000 \
    --gemb-steps 100000 \
    --print-interval 100 \
    --save-interval 300 \
    --early-stop \
    --patience 5
