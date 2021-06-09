import numpy as np

# 로스 및 메트릭 출력 템플릿
templet = '>>> Loss {0:0.5f} Depth metric | a1 {1:0.5f} | a2 {2:0.5f} | a3 {3:0.5f} | rmse {4:0.5f} | rmse_log {5:0.5f} | abs_rel {6:0.5f} | sqrt_rel {7:0.5f}'


def model_print(epoch, train_log, valid_loss):
    print("EPOCH   {0}".format(epoch+1))
    print(templet.format(
        np.mean(train_log["loss"]),
        np.mean(train_log["a1"]),
        np.mean(train_log["a2"]),
        np.mean(train_log["a3"]), 
        np.mean(train_log["rmse"]), 
        np.mean(train_log["rmse_log"]), 
        np.mean(train_log["abs_rel"]), 
        np.mean(train_log["sq_rel"])))
    print(templet.format(
        np.mean(valid_loss["loss"]),
        np.mean(valid_loss["a1"]),
        np.mean(valid_loss["a2"]),
        np.mean(valid_loss["a3"]), 
        np.mean(valid_loss["rmse"]), 
        np.mean(valid_loss["rmse_log"]), 
        np.mean(valid_loss["abs_rel"]), 
        np.mean(valid_loss["sq_rel"])))