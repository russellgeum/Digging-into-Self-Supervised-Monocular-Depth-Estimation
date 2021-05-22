# 로스 및 메트릭 출력 템플릿
templet = '>>> Total Loss {0:0.5f} Depth metric | a1 {1:0.5f} | a2 {2:0.5f} | a3 {3:0.5f} | rmse {4:0.5f} | rmse_log {5:0.5f} | abs_rel {6:0.5f} | sqrt_rel {7:0.5f}'
def logger(epoch, train_log, valid_loss):
        print("EPOCH   {0}".format(epoch))
        print(templet.format(
                train_log["loss"],
                train_log["a1"],
                train_log["a2"],
                train_log["a3"], 
                train_log["rmse"], 
                train_log["rmse_log"], 
                train_log["abs_rel"], 
                train_log["sq_rel"]))
        if valid_loss is not None:
                print(templet.format(
                        valid_loss["loss"],
                        valid_loss["a1"],
                        valid_loss["a2"],
                        valid_loss["a3"], 
                        valid_loss["rmse"], 
                        valid_loss["rmse_log"], 
                        valid_loss["abs_rel"], 
                        valid_loss["sq_rel"]))
        else:
                pass