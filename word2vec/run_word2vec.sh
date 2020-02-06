python word2vec.py \
    --data_dir ./worddata/text8.train.txt \
    --batch_size 16 \
    --do_train \
    --max_vocab_size 30000 \
    --embed_size 200 \
    --C 5   \
    --k 100 \
    --num_epoch 5 \
    --learnning_rate 5e-4