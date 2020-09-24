from __future__ import division
import os
import numpy as np
import PIL.Image as pil
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from pylab import *

from SfMLearner import SfMLearner
from utils import normalize_depth_for_display

# added by ZYD
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# adding end

img_height=128
img_width=416
ckpt_file = 'models/kitti_depth_model/model-190532'
fh = open('misc/sample.png', 'rb')
I = pil.open(fh)
I = I.resize((img_width, img_height), pil.ANTIALIAS)
I = np.array(I)

sfm = SfMLearner()
sfm.setup_inference(img_height,
img_width,
mode='depth')

saver = tf.train.Saver([var for var in tf.model_variables()]) 
with tf.Session() as sess:
	saver.restore(sess, ckpt_file)
	pred = sfm.inference(I[None,:,:,:], sess, mode='depth')

plt.figure()
plt.subplot(121)
cv2.imshow("source",I)
plt.subplot(122)
cv2.imshow("result",normalize_depth_for_display(pred['depth'][0,:,:,0]))
cv2.waitKey()
