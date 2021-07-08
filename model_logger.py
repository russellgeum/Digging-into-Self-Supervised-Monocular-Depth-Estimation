import os
import numpy as np
import torch


# 로스 및 메트릭 출력 템플릿
def model_print(epoch, metric, train_log, valid_log):
    print(">>>   EPOCH   {}".format(epoch+1))
    print(">>>   Train Log")
    for index, key in enumerate(metric):
        if index != len(metric) - 1:
            print("{}: {:0.3f}   ".format(key, np.mean(train_log[key])), end = ' ')
        else:
            print("{}: {:0.3f}   ".format(key, np.mean(train_log[key])))

    print(">>>   Valid Log")
    for index, key in enumerate(metric):
        if index != len(metric) - 1:
            print("{}: {:0.3f}   ".format(key, np.mean(valid_log[key])), end = ' ')
        else:
            print("{}: {:0.3f}   ".format(key, np.mean(valid_log[key])))