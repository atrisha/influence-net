'''
Created on Feb 8, 2017

@author: atri
'''
import nn_train
import sys
import time
import tensorflow as tf

results = []
for i in range(20):
    res_str = nn_train.main(None)
    results.append(res_str)
    time.sleep(2)

for res in results:
    print(res)