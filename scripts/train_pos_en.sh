python codes/main.py \
    --use-gemb \
    --config-file models/pos-en-config.pkl \
    --train-data-file data/pos_en/train.pkl \
    --dev-data-file data/pos_en/dev.pkl \
    --test-data-file data/pos_en/test.pkl \
    --task tagging \
    --keep-prob 0.9 \
    --dictfile data/pos_en/dicts.pkl \
    --embed-dim 300 \
    --hidden-dims 256 256 \
    --ckpt models/pos-en-model \
    --max-ckpts 20 \
    --batch-size 32 \
    --max-steps 1000000 \
    --gemb-steps 1000000 \
    --print-interval 50 \
    --save-interval 1000
