import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import h5py
import IPython.display as display
from keras.layers import Dense,Flatten,Dropout
import os
print(os.getcwd())


class SFContourEmbedder:
    def __init__(self,checkpoint='pitchVocalSetLatentSlot.ckpt'):
        self.img_input1 = tf.placeholder(tf.float32, shape=[None,100])
        self.img_input2 = tf.placeholder(tf.float32, shape=[None,100])
        self.img_input3 = tf.placeholder(tf.float32, shape=[None,100])
        self.learning_rate = tf.placeholder(tf.float32)

        self.e1=self.getEmb(self.img_input1)
        self.e3=self.getEmb(self.img_input3)

        self.e2=self.e3-self.e1

        self.yop1=self.Decoder(self.e1)
        self.yop2=self.Decoder(self.e2)
        self.yop3=self.Decoder(self.e3)
        self.mse1 = tf.reduce_sum(tf.abs(self.yop1-self.img_input1))
        self.mse2 = tf.reduce_sum(tf.abs(self.yop2-self.img_input2))
        self.mse3 = tf.reduce_sum(tf.abs(self.yop3-self.img_input3))
        print(self.yop1)
        print(self.yop2)
        print(self.yop3)


        self.mse = self.mse2+self.mse1+self.mse3
        self.learning_rate = tf.placeholder(tf.float32)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mse)
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, checkpoint)

    def getEmb(self,xx):
        f1 = Dense(2048,activation='relu')(xx)
        #f1=Dropout(0.2)(f1)
        f2 = Dense(2048,activation='relu')(f1)
        #f2=Dropout(0.2)(f2)
        f3 = Dense(20,activation='relu')(f2)
        return f3

    def Decoder(self,currentex):
        d1 = Dense(2048,activation='relu')(currentex)
        #d1=Dropout(0.2)(d1)
        d2 = Dense(2048,activation='relu')(d1)
        #d2=Dropout(0.2)(d2)
        d3 = Dense(100,activation=None)(d2)
        return d3
        
    def embedContour(self,contour):
        contour = contour.reshape(1,100)
        E=self.sess.run(self.e1, feed_dict={self.img_input1:contour}) # Get the embedding
        return E[0]
