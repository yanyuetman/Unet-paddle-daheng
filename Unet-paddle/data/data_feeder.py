import os
import cv2
import numpy as np
from paddle.fluid import core

# Feed Data into Tensor
def get_feeder_data(data, place, for_test=False):
    feed_dict = {}
    image_t = core.LoDTensor()
    image_t.set(data[0], place)
    feed_dict["image"] = image_t

    # if not test, feed label also
    # Otherwise, only feed image
    if not for_test:
        labels_t = core.LoDTensor()
        labels_t.set(data[1], place)
        feed_dict["label"] = labels_t
    return feed_dict

def get_feeder_test_data(data, place, for_test=False):
    feed_dict = {}
    image_t = core.LoDTensor()
    image_t.set(data, place)
    feed_dict["image"] = image_t

    # if not test, feed label also
    # Otherwise, only feed image
    return feed_dict




# Train Images Generator
def train_image_gen(train_list, batch_size=4, image_size=[256, 256], crop_offset=0):
    # Arrange all indexes
    all_batches_index = np.arange(0, len(train_list['image']))
    out_images = []
    out_masks = []
    image_dir = np.array(train_list['image'])
    label_dir = np.array(train_list['label'])
    np.random.shuffle(all_batches_index)
    print(label_dir[0])
    print('index',len(all_batches_index))
    while True:
        np.random.shuffle(all_batches_index)
        # Random shuffle indexes every epoch
        for index in all_batches_index:
            if os.path.exists(image_dir[index]):
                train_image = cv2.imread(image_dir[index])
                train_image = cv2.resize(train_image, (image_size[0],image_size[1]), interpolation=cv2.INTER_NEAREST)
		train_mask = cv2.imread(label_dir[index], cv2.IMREAD_GRAYSCALE)
		train_mask = cv2.resize(train_mask, (image_size[0], image_size[1]), interpolation=cv2.INTER_NEAREST)
                # Crop the top part of the image
                # Resize to train size
                #train_img, train_mask = crop_resize_data(ori_image, ori_mask, image_size, crop_offset)
                # Encode
                #train_mask = encode_labels(train_mask)

                # verify_labels(train_mask)
                out_images.append(train_image)
                out_masks.append(train_mask)
                if len(out_images) >= batch_size:
                    out_images = np.array(out_images)
                    out_masks = np.array(out_masks)
                    out_images = (out_images[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) - 128) / 128.0
                    out_masks = out_masks.astype(np.int64)
                    yield out_images, out_masks
                    out_images, out_masks = [], []
            else:
                print(image_dir, 'does not exist.')

# Validation Images Generator
def val_image_gen(val_list, batch_size=4, image_size=[256, 256], crop_offset=0):
    all_batches_index = np.arange(0, len(val_list['image']))

    out_images = []
    out_masks = []
    image_name = []
    image_dir = np.array(val_list['image'])
    label_dir = np.array(val_list['label'])
    np.random.shuffle(all_batches_index)
    while True:
        #np.random.shuffle(all_batches_index)
        for index in all_batches_index:
            if os.path.exists(image_dir[index]):
                val_image = cv2.imread(image_dir[index])
                val_image = cv2.resize(val_image, (image_size[0],image_size[1]), interpolation=cv2.INTER_NEAREST)
                val_mask = cv2.imread(label_dir[index], cv2.IMREAD_GRAYSCALE)
		val_mask = cv2.resize(val_mask, (image_size[0], image_size[1]), interpolation=cv2.INTER_NEAREST)
                out_images.append(val_image)
                out_masks.append(val_mask)
                image_name.append(image_dir[index])
                if len(out_images) >= batch_size:
                    out_images = np.array(out_images)
                    out_masks = np.array(out_masks)
                    out_images = (out_images[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) - 128) / 128.0
                    out_masks = out_masks.astype(np.int64)
                    yield out_images, out_masks, image_name
                    out_images, out_masks = [], []
                    image_name = []
            else:
                print(image_dir, 'does not exist.')

def test_image_gen(test_list, batch_size=4, image_size=[512, 512], crop_offset=0):
    all_batches_index = np.arange(0, len(test_list['image']))
    out_images = []
    image_dir = np.array(test_list['image'])
    while True:
        for index in all_batches_index:
            if os.path.exists(image_dir[index]):
                test_image = cv2.imread(image_dir[index])
                out_images.append(test_image)
                if len(out_images) >= batch_size:
                    out_images = np.array(out_images)
                    out_images = (out_images[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) - 128) / 128.0
                    yield out_images
                    out_images  = []
            else:
                print(image_dir, 'does not exist.')


