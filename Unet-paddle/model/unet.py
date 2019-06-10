from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.initializer import Normal
from paddle.fluid.param_attr import ParamAttr
import math
__all__ = ['UNet']
train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}



class UNet():
	def __init__(self):
		self.params = train_parameters
	def net(self, input, class_num, test=False):
		conv1 = self.encoder_block(input, 64, pool=True, dropout=None, is_test=test)
		conv_1 = fluid.layers.pool2d(conv1, pool_size=2, pool_type='max', pool_stride=2)#128*128
		conv2 = self.encoder_block(conv_1, 128, pool=True, dropout=None, is_test=test)
		conv_2 = fluid.layers.pool2d(conv2, pool_size=2, pool_type='max', pool_stride=2)#64*64
		conv3 = self.encoder_block(conv_2, 256, pool=True, dropout=None, is_test=test)
		conv_3 = fluid.layers.pool2d(conv3, pool_size=2, pool_type='max', pool_stride=2)#32*32
		conv4 = self.encoder_block(conv_3, 512, pool=True, dropout=None, is_test=test) #dropout
		conv_4 = fluid.layers.pool2d(conv4, pool_size=2, pool_type='max', pool_stride=2)#16*16
		conv5 = self.encoder_block(conv_4, 1024, pool=None, dropout=None, is_test=test) # 16*16 #dropout
		deconv_1 = self.decoder_block(conv5, conv4, 512) #32*32 
		deconv_2 = self.decoder_block(deconv_1, conv3, 256)#64*64
		deconv_3 = self.decoder_block(deconv_2, conv2, 128)#128*128
		deconv_4 = self.decoder_block(deconv_3, conv1, 64)#256*256
		conv6 = self.conv_layer(deconv_4, num_filters = class_num, filter_size = 3, act = 'relu')
		out = self.conv_layer(conv6, num_filters = class_num, filter_size = 1, act='relu')
		return out

	def conv_bn_layer(self, input, num_filters, filter_size, stride=1, groups=1, act=None, bn=False, bias_attr=False):
		conv = fluid.layers.conv2d(
			input=input,
			num_filters=num_filters,
			filter_size=filter_size,
			stride=stride,
			padding=(filter_size - 1) // 2,
			groups=groups,
			act=act,
			bias_attr=bias_attr,
			param_attr=ParamAttr(initializer=MSRA()))
		if bn == True:
			conv = fluid.layers.batch_norm(input=conv, act=act)
		return conv
	def conv_layer(self, input, num_filters, filter_size, stride=1, groups=1, act=None, bias_attr=False):
		conv = fluid.layers.conv2d(
			input=input,
			num_filters=num_filters,
			filter_size=filter_size,
			stride=stride,
			padding=(filter_size - 1) // 2,
			groups=groups,
			act=act,
			bias_attr=bias_attr,
			param_attr=ParamAttr(initializer=MSRA()))
		return conv
	def encoder_block(self, input, num_filters, pool=True, dropout=None, is_test=False):
		conv_bn = self.conv_layer(input=input, num_filters=num_filters, filter_size=3, act='relu')
		conv_bn = self.conv_layer(input=conv_bn, num_filters=num_filters, filter_size=3, act='relu')
		if dropout == True:
	           conv_bn == fluid.layers.dropout(conv_bn, 0.3, is_test)
		return conv_bn
	def decoder_block(self, input, concate_input, num_filters):
		deconv = input
		#deconv =  fluid.layers.resize_bilinear(deconv, out_shape=(deconv.shape[2] * 2, deconv.shape[3] * 2))
		deconv = self.conv_layer(input=deconv, num_filters=num_filters, filter_size=3, act='relu')
		deconv =  fluid.layers.resize_bilinear(deconv, out_shape=(deconv.shape[2] * 2, deconv.shape[3] * 2))
		merge = fluid.layers.concat([deconv, concate_input], axis=1)
		deconv_bn = self.conv_layer(input=merge, num_filters=num_filters, filter_size=3, act='relu')
		deconv_bn = self.conv_layer(input=deconv_bn, num_filters=num_filters, filter_size=3, act='relu')
		return deconv_bn
	
	
		





		        
	
        

