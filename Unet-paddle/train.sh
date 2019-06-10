#!/bin/sh
python ./daheng_train.py \
    -train_file /home/mengxinyue/daheng/train_list.txt \
    -valid_file /home/mengxinyue/daheng/test_list.txt \
    -save_model /home/mengxinyue/daheng_model/ \
    -save_infer /home/mengxinyue/daheng_infer/ \
    -pretrain_model /home/mengxinyue/daheng_model/best_end/ \
    -data_dir /home/mengxinyue/daheng/
