import sys
import scipy
import matplotlib.pyplot as plt
from keras import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import h5py
import IPython.display as display
from keras.layers import Dense,Flatten
import os
print(os.getcwd())

class PTContourEmbedder:
    def __init__(self, checkpoint='pitchVocalSet.ckpt'):
        self.img_input1 = tf.placeholder(tf.float32, shape=[None,1,100,1])
        self.img_input2 = tf.placeholder(tf.float32, shape=[None,1,100,1])
        self.yref = tf.placeholder(tf.float32, shape=[None,2])
        self.learning_rate = tf.placeholder(tf.float32)
        self.e1=self.getEmb(self.img_input1)
        self.e2=self.getEmb(self.img_input2)
        self.e=tf.concat([self.e1,self.e2],axis=1)
        self.e = Dense(256,activation='relu')(self.e)
        self.yop=Dense(2,activation=None)(self.e)

        self.mse = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.yref, logits=self.yop))
        self.learning_rate = tf.placeholder(tf.float32)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mse)
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, checkpoint)

    def getEmb(self,img_input):
        x = layers.Conv2D(64, (1, 3),
                              activation='relu',
                              padding='same',
                              name='block1_conv1')(img_input)
        x = layers.Conv2D(64, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv2')(x)
        x = layers.MaxPooling2D((1, 2), strides=(1, 2), name='block1_pool')(x)



        x = layers.Conv2D(128, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv1')(x)
        x = layers.Conv2D(128, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv2')(x)
        x = layers.MaxPooling2D((1, 2), strides=(1,2), name='block2_pool')(x)

        x = layers.Conv2D(256, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv1')(x)
        x = layers.Conv2D(256, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv2')(x)
        x = layers.Conv2D(256, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv3')(x)
        x = layers.Conv2D(256, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv4')(x)
        x = layers.MaxPooling2D((1, 2), strides=(1, 2), name='block3_pool')(x)


        x = layers.Conv2D(512, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv1')(x)
        x = layers.Conv2D(512, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv2')(x)
        x = layers.Conv2D(512, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv3')(x)
        x = layers.Conv2D(512, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv4')(x)
        x = layers.MaxPooling2D((1, 2), strides=(1, 2), name='block4_pool')(x)

        x = layers.Conv2D(256, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv1')(x)
        x = layers.Conv2D(128, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv2')(x)
        x = layers.Conv2D(128, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv3')(x)
        x = layers.Conv2D(128, (1, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv4')(x)
        x = layers.MaxPooling2D((1, 2), strides=(1, 2), name='block5_pool')(x)
        # Block 2
        print(x.shape)
        x = Flatten()(x)
        print(x)
        x = Dense(128,activation='relu')(x)
        print(x)
        return x

    def embedContour(self,contour):
        contour = contour.reshape(1,1,100,1)
        E=self.sess.run(self.e1, feed_dict={self.img_input1:contour}) # Get the embedding
        return E[0]
