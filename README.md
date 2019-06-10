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
    -train_file ./daheng/train_list.txt \
    -valid_file ./test_list.txt \
    -save_model ./daheng_model (save model param path)\
    -save_infer ./daheng_infer (save infer model path) \
    -pretrain_model ./best_end (pretrain model path) \
    -data_dir ./daheng/ 


# Test
CUDA_VISIBLE_DEVICES=0 sh test.sh

python ./daheng_test.py \
    -test_file ./daheng/test_list.txt \
    -model_path ./best_end (best model path for predict) \
    -data_dir ./daheng/ \
    -pre_savedir ./save_pre_image/ 

