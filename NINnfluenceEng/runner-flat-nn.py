'''
Created on Feb 8, 2017

@author: atri
'''
import flat_nn_train
import sys

results = []
for i in range(20):
    res_str = flat_nn_train.main(sys.argv[0])
    results.append(res_str)

for res in results:
    print(res)