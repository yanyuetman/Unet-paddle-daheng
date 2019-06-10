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
    return fluid.layers.reduce_mean(loss), miou

def create_network(train_image, train_label, classes, image_size=(512, 512), for_test=False):
    Net = model.unet.UNet()
    predict = Net.net(train_image, classes, test=for_test)
    
    #predict = unet_simple(train_image, classes, image_size)
    #print('The program will run', network)
    if for_test == False:
        loss, miou = create_loss(predict, train_label, classes)
        return loss, miou, predict
    elif for_test == True:
        return predict, miou
    else:
        raise Exception('Wrong Status:', for_test)

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
    parser.add_argument('-train_file', default=None)
    parser.add_argument('-valid_file', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_infer', default=None)
    parser.add_argument('-pretrain_model', default=None)
    parser.add_argument('-data_dir', default=None)
    args = parser.parse_args()
    return args




# The main method
def main():
    IMG_SIZE =[224, 224]
    add_num = 13
    num_classes = 8
    batch_size = 8
    log_iters = 1
    base_lr = 0.0001
    valid_batch_size =  1
    save_model_iters = 50
    use_pretrained = True
    network = 'unet'
    args = get_args()
    save_model_path = args.save_model
    save_infer_path = args.save_infer
    model_path = args.pretrain_model
    epoches = 400
    crop_offset = 0
    data_dir = args.data_dir
    daheng_train_file = args.train_file
    daheng_valid_file = args.valid_file
    train_file = open(daheng_train_file,'r')
    valid_file = open(daheng_valid_file,'r')
    train_lines = [line.strip('\n').split('\t')[0].replace('./', data_dir)for line in train_file ]
    val_lines = [line.strip('\n').split('\t')[0].replace('./', data_dir) for line in valid_file ]
    random.shuffle(train_lines)
    random.shuffle(val_lines)
    label_lines = [item.replace('images','labels').replace('.jpg','.png') for item in train_lines]
    val_label_lines = [item.replace('images','labels').replace('.jpg','.png') for item in val_lines]
    #val_lines = [os.path.join(val_dir, item) for item in val_list if item.find('image') != -1]
    #val_label_lines = [item.replace('image','mask') for item in val_lines]
    #training set input
    train_list = {}
    train_list['image'] = train_lines
    train_list['label'] = label_lines
    #validation set input
    valid_list = {}
    valid_list['image'] = val_lines
    valid_list['label'] = val_label_lines
    print(valid_list['image'][0])
    print(valid_list['label'][0])
    
    # Get data list and split it into train and validation set.

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
    # Train
    #print('Train Data Size:', len(train_lines))
    train_reader = train_image_gen(train_list, batch_size, IMG_SIZE, crop_offset)
    valid_reader = val_image_gen(valid_list, batch_size, IMG_SIZE, crop_offset)
    # Create model and define optimizer
    reduced_loss, miou, pred = create_network(images, labels, num_classes, image_size=(IMG_SIZE[1], IMG_SIZE[0]), for_test=False)
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=base_lr)
    optimizer.minimize(reduced_loss, no_grad_set=no_grad_set)
    main_program = fluid.default_main_program()
    test_program = fluid.default_main_program().clone(for_test=True)
    # Whether load pretrained model
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    if use_pretrained == True:
        fluid.io.load_params(exe, model_path)
        print("loaded model from: %s" % model_path)
    else:
        print("Train from initialized model.")
    # Parallel Executor to use multi-GPUs
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.allow_op_delay = True
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
    train_exe = fluid.ParallelExecutor(main_program=main_program, use_cuda=True, loss_name=reduced_loss.name,
                                       build_strategy=build_strategy, exec_strategy=exec_strategy)
    # Training
    for epoch in range(epoches):
        print('Start Training Epoch: %d'%(epoch + 1))
        train_length = len(train_lines)
        for iteration in range(int(train_length / batch_size)):
            iter_id +=1
            train_data = next(train_reader)
            results = train_exe.run(
                feed=get_feeder_data(train_data, place),
                fetch_list=[reduced_loss.name, miou.name, pred.name])
             
            iter_id += 1
            total_loss += np.mean(results[0])
            total_miou += np.mean(results[1])
            if iter_id % log_iters == 0: # Print log
                end_time = time.time()
                print(
                "Iter - %d: train loss: %.3f, mean iou: %.3f, time cost: %.3f s"
                % (iter_id, total_loss / log_iters, total_miou / log_iters, end_time - prev_time))
                total_loss = 0.0
                total_miou = 0.0
                prev_time = time.time()
            if iter_id % 150 == 0:
               dir_name = save_model_path + str(epoch) + str(iter_id) + '_end'
               fluid.io.save_params(exe, dirname=dir_name) 
#evluation on validation_set and save the best model
            if iter_id % 50 == 0:
                    it = 0 
                    #for iter in range(int(len(valid_list['image']) / valid_batch_size)):
                    print(len(valid_list['image']))
                    for iter in range(0, 50):
                        it +=1
                        valid_data = next(valid_reader)
                        results = exe.run(test_program,
                            feed=get_feeder_data(valid_data, place),
                            fetch_list=[pred.name, miou.name])
                        total_miou +=np.mean(results[1])
                        prediction = np.argmax(results[0][0], axis=0)
                        submission_mask = prediction * 255
                        print(np.mean(results[1]))
                        #cv2.imwrite(os.path.join(sub_dir, str(iter)+'.png'), submission_mask)

                    print(total_miou / it)
                    #event_handler_plot(test_prompt, iter_id, (total_miou / it))
                    if (total_miou / it) > old_miou:
                       old_miou = total_miou / it
                       dir_name =save_model_path + 'best_end'
                       shutil.rmtree(dir_name,ignore_errors=True)
                       os.makedirs(dir_name)
                       fluid.io.save_params(exe, dirname=dir_name)
                       print("Saved best checkpoint: %s" % (dir_name))
                       save_path = save_infer_path + 'best_end'
                       shutil.rmtree(save_path,ignore_errors=True)
                       os.makedirs(save_path)
                       fluid.io.save_inference_model(save_path,feeded_var_names=[images.name],target_vars=[umodel],executor=exe)
                       print("Saved best infer checkpoint: %s" % (save_path))
                

# Main
if __name__ == "__main__":
    main()
