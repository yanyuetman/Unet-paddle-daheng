# Unet-paddle-daheng
Daheng dataset Segmentation  Task implement on paddlepaddle

# Data_files
├── daheng                      //
│   ├── images
│          ├──白点
│          ├──黑点
│          ├──漏涂
│          ├──异物
│          ├──小白线
│          ├──烧伤
│          ├──孔或透明
│          ├──其他
│        
│   ├── labels
│          ├──白点
│          ├──黑点
│          ├──漏涂
│          ├──异物
│          ├──小白线
│          ├──烧伤
│          ├──孔或透明
│          ├──其他
│        
│   ├── Unet-paddle         // 实验     

# Train
CUDA_VISIBLE_DEVICES=0,1,2 sh train.sh

# Test
CUDA_VISIBLE_DEVICES=0,1,2 sh test.sh
