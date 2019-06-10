# Unet-paddle-daheng
Daheng dataset Segmentation  Task implement on paddlepaddle
```
.
|-- Unet-paddle
|   |-- daheng_best_model
|   |   `-- best_end
|   |-- daheng_test.py
|   |-- daheng_train.py
|   |-- data
|   |   |-- __init__.py
|   |   |-- data_feeder.py
|   |   `-- reader.py
|   |-- model
|   |   |-- __init__.py
|   |   `-- unet.py
|   |-- test.sh
|   `-- train.sh
`-- daheng
    |-- images
    |   |-- 其他
    |   |-- 异物
    |   |-- 漏涂
    |   |-- 烧伤
    |   |-- 白点
    |   |-- 黑点
    |   |-- 小白线
    |   `-- 孔或透明
    |-- labels
    |   |-- 其他
    |   |-- 异物
    |   |-- 漏涂
    |   |-- 烧伤
    |   |-- 白点
    |   |-- 黑点
    |   |-- 小白线
    |   `-- 孔或透明
    |-- test_list.txt
    `-- train_list.txt
```

# Train
CUDA_VISIBLE_DEVICES=0,1,2 sh train.sh

python ./daheng_train.py \
    -train_file /home/mengxinyue/daheng/train_list.txt \
    -valid_file /home/mengxinyue/daheng/test_list.txt \
    -save_model /home/mengxinyue/daheng_model/ \
    -save_infer /home/mengxinyue/daheng_infer/ \
    -pretrain_model /home/mengxinyue/daheng_model/best_end/ \
    -data_dir /home/mengxinyue/daheng/


# Test
CUDA_VISIBLE_DEVICES=0,1,2 sh test.sh
