python codes/main.py \
    --train-gemb false \
    --config-file ../models/pos-en-config.pkl \
    --train-data-file ../data/pos_en/train.txt \
    --dev-data-file ../data/pos_en/dev.txt \
    --test-data-file ../data/pos_en/test.txt \
    --task tagging \
    --keep-prob 0.9 \
    --dictfile ../data/pos_en/dicts.pkl \
    --embed-dim 300 \
    --hidden-dims 256 256 \
    --ckpt ../models/pos-en-model \
    --max-ckpts 20 \
    --batch-size 64 \
    --max-steps 1000000 \
    --gemb-steps 1000000 \
    --print-interval 50 \
    --save-interval 1000