# Unet-paddle-daheng
Daheng dataset Segmentation  Task implement on paddlepaddle
```
.
|-- 1.txt
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

# Test
CUDA_VISIBLE_DEVICES=0,1,2 sh test.sh
