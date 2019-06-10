import time
import cv2
import sys, argparse
import numpy as np
import paddle.fluid as fluid
import random
from data.data_feeder import get_feeder_data, train_image_gen, val_image_gen
import  model.unet
import os
import shutil
from model.unet_base import unet_base
from model.unet_simple import unet_simple
from paddle.utils.plot import Ploter
# Compute Mean Iou
def mean_iou(pred, label, num_classes):
    pred = fluid.layers.argmax(pred, axis=1)
    pred = fluid.layers.cast(pred, 'int32')
    label = fluid.layers.cast(label, 'int32')
    miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes)
    return miou

# Get Loss Function
no_grad_set = []
def create_loss(predict, label, num_classes):
    predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
    predict = fluid.layers.reshape(predict, shape=[-1, num_classes])
    predict = fluid.layers.softmax(predict)
    label = fluid.layers.reshape(label, shape=[-1, 1])
    # BCE with DICE
    bce_loss = fluid.layers.cross_entropy(predict, label)
    dice_loss = fluid.layers.dice_loss(predict, label)
    no_grad_set.append(label.name)
    #loss = bce_loss + dice_loss
    loss = bce_loss
    miou = mean_iou(predict, label, num_classes)
    return miou

def create_network(train_image, train_label, classes, image_size=(512, 512), for_test=False):
    Net = model.unet.UNet()
    predict = Net.net(train_image, classes, test=for_test)
    
    #predict = unet_simple(train_image, classes, image_size)
    #print('The program will run', network)
    miou = create_loss(predict, train_label, classes)
    return predict, miou

def event_handler(pass_id, batch_id, miou):
    print("Pass %d, cost %d, miou %f" % (pass_id, cost, miou))

def event_handler_plot(ploter_title, step, miou):
    train_prompt = "Train cost"
    test_prompt = "Test cost"
    cost_ploter = Ploter(train_prompt, test_prompt)
    cost_ploter.append(ploter_title, step, miou)
    cost_ploter.plot()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_file', default=None)
    parser.add_argument('-model_path', default=None)
    parser.add_argument('-data_dir', default=None)
    parser.add_argument('-pre_savedir', default='/home/mengxinyue/Unet-paddle/dataset/daheng_pre/image')
    args = parser.parse_args()
    return args





# The main method
def main():
    IMG_SIZE =[224, 224]
    add_num = 13
    num_classes = 8
    batch_size = 1
    log_iters = 1
    base_lr = 0.0001
    valid_batch_size =  1
    save_model_iters = 50
    use_pretrained = True
    network = 'unet'
    args = get_args()
    model_path = args.model_path
    epoches = 400
    crop_offset = 0
    data_dir = args.data_dir
    daheng_valid_file = args.test_file
    sub_dir = args.pre_savedir
    valid_file = open(daheng_valid_file,'r')
    val_lines = [line.strip('\n').split('\t')[0].replace('./',data_dir) for line in valid_file ]
    val_label_lines = [item.replace('images','labels').replace('.jpg','.png') for item in val_lines]
    #validation set input
    valid_list = {}
    valid_list['image'] = val_lines
    valid_list['label'] = val_label_lines
    

    #Initialization
    images = fluid.layers.data(name='image', shape=[3, IMG_SIZE[1], IMG_SIZE[0]], dtype='float32')
    labels = fluid.layers.data(name='label', shape=[1, IMG_SIZE[1], IMG_SIZE[0]], dtype='float32')
    
    #model definition for inference save 
    Net = model.unet.UNet()
    umodel = Net.net(images, num_classes, test=True)

    iter_id = 0
    total_loss = 0.0
    total_miou = 0.0
    val_miou = 0.0
    old_miou = 0.0
    train_prompt = "Train cost"
    test_prompt = "Test cost" 
    prev_time = time.time()
    valid_reader = val_image_gen(valid_list, batch_size, IMG_SIZE, crop_offset)
    # Create model and define optimizer
    pred, miou = create_network(images, labels, num_classes, image_size=(IMG_SIZE[1], IMG_SIZE[0]), for_test=True)
    main_program = fluid.default_main_program()
    test_program = fluid.default_main_program().clone(for_test=True)
    # Whether load pretrained model
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    fluid.io.load_params(exe, model_path)
    # Parallel Executor to use multi-GPUs
    """
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.allow_op_delay = True
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reducecd
    train_exe = fluid.ParallelExecutor(main_program=main_program, use_cuda=True,
                                       build_strategy=build_strategy, exec_strategy=exec_strategy)
    """
    it = 0
    # Test
    print(len(valid_list['image']))
    for iter in range(0, len(valid_list['image'])):
             it +=1
             image, label, name = next(valid_reader)
             valid_data = (image, label)
             print(name[0])
             save_name = name[0].split('/')
             save_dir = os.path.join(sub_dir,save_name[5])
             results = exe.run(test_program,
                          feed=get_feeder_data(valid_data, place),
                          fetch_list=[pred.name, miou.name])
             total_miou +=np.mean(results[1])
             prediction = np.argmax(results[0][0], axis=0)
             print(np.mean(results[1]))
             #cv2.imwrite(os.path.join(save_dir, save_name[-1].replace('jpg','png')), prediction) save prediction images
    print(total_miou / it)
                

# Main
if __name__ == "__main__":
    main()
