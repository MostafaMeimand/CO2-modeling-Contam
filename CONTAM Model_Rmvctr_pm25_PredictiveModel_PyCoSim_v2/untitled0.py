#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:15:29 2022

@author: ranakhier
"""

import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Convolution2D as Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from collections import deque
import time
import random
import tensorflow as tf

def discounted_reward(reward, i):
    FACTOR = 0.999
    if reward <= 0:
        return reward / np.power(FACTOR, i)
    else:
        return reward * np.power(FACTOR, i)