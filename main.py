import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict as cvp
import random
from functools import reduce
from collections import Counter
import cv2
import pickle
from gcforest import *
#https://github.com/STO-OTZ/my_gcForest/blob/master/my_gcForest.ipynb
 
with open('./train.pickle','rb') as f:
    train_features,train_labels,test_features,test_labels = pickle.load(f)
x_train = train_features[:,:,:,0:4]
y_train = train_labels.reshape([-1,])
x_test = test_features[:,:,:,0:4]
y_test = test_labels.reshape([-1,])
print('数据集信息如下:')
print('训练集的维度.....................:', x_train.shape)
print('训练集标签的维度..................:', y_train.shape)
print('测试集的维度.....................:', x_test.shape)
print('测试集标签的维度..................:', y_test.shape)

scan_forest_params1 = RandomForestClassifier(n_estimators=30, min_samples_split=21, max_features=1,n_jobs=-1).get_params()
scan_forest_params2 = RandomForestClassifier(n_estimators=30, min_samples_split=21, max_features='sqrt',n_jobs=-1).get_params()
cascade_forest_params1 = RandomForestClassifier(n_estimators=100, min_samples_split=11, max_features=1,n_jobs=-1).get_params()
cascade_forest_params2 = RandomForestClassifier(n_estimators=100, min_samples_split=11, max_features='sqrt',n_jobs=-1).get_params()
scan_params_list = [scan_forest_params1,  scan_forest_params2]
cascade_params_list = [cascade_forest_params1, cascade_forest_params2] * 2

def calc_accuracy(pre,y):
    return float(sum(pre==y))/len(y)

class ProbRandomForestClassifier(RandomForestClassifier):
    def predict(self, X):
        return RandomForestClassifier.predict_proba(self, X)

train_size = x_train.shape[0]
# gcForest 

# Multi-Grained Scan Step
import time
print("多粒度扫描开始:")
scan_time_start = time.time()
Scaner1 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./4)
Scaner2 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./2)
Scaner3 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./16)

###########################################################################################
x_train_scan =np.hstack([scaner.scan_fit(x_train[:train_size].reshape((train_size,28,28,4)), y_train[:train_size]) for scaner in [Scaner2][:1]])
###########################################################################################

scan_time_end = time.time()
print('多粒度扫描完成..........:\n')
print('多粒度扫描时间..........:',scan_time_end - scan_time_start,'s')
# X_test_scan = np.hstack([scaner.scan_predict(X_test.reshape((len(X_test),28,28)))
#                              for scaner in [Scaner1,Scaner2,Scaner3][:1]])

# Cascade RandomForest Step
print("级联森林训练开始:")
CascadeRF = CascadeForest(ProbRandomForestClassifier(),cascade_params_list)
start_time = time.time()
CascadeRF.fit(x_train_scan, y_train[:train_size])
end_time = time.time()
print("级联森林训练结束:")
# y_pre_staged = CascadeRF.predict_staged(X_test_scan)
# test_accuracy_staged = np.apply_along_axis(lambda y_pre: calc_accuracy(y_pre,y_test), 1, y_pre_staged)
# print('\n'.join('level {}, test accuracy: {}'.format(i+1,test_accuracy_staged[i]) for i in range(len(test_accuracy_staged))))
print('级联森林训练耗费时间是:',end_time - start_time,'s')


pred_time_start = time.time()
with open('./502_561_4.pickle','rb') as f:
    x_test = pickle.load(f)
x_test = x_test[:,:,:,0:4]
prediction = []

###########################################################################################
for i in range(28):
    x_test_scan = np.hstack([scaner.scan_predict(x_test[i * 10000:(i + 1) * 10000].reshape((-1,28,28,4))) for scaner in [Scaner2][:1]])
    y_pre_staged_1 = CascadeRF.predict_staged(x_test_scan)
    prediction.append(y_pre_staged_1[-1])
x_test_scan = np.hstack([scaner.scan_predict(x_test[280000:].reshape((-1,28,28,4)))for scaner in [Scaner2][:1]])
###########################################################################################

y_pre_staged_2 = CascadeRF.predict_staged(x_test_scan)
for i in range(28):
    if i == 0:
        pred_1 = prediction[0].reshape([-1,])
    else:
        pred_1 = np.concatenate([pred_1,prediction[i].reshape([-1,])])
        
pred = np.concatenate([pred_1,y_pre_staged_2[-1].reshape([-1,])],axis = 0)
print("Counter(pred) = ",Counter(pred))
print("pred.shape = ",pred.shape)

pred_image = pred.reshape([502,561])
img = np.zeros([502,561,3])
for i in range(pred_image.shape[0]):
    for j in range(pred_image.shape[1]):
        if pred_image[i][j] == 0:
            img[i][j][0] = 255        #红色
            img[i][j][1] = 0
            img[i][j][2] = 0
        elif pred_image[i][j] == 1:   #蓝色
            img[i][j][0] = 0
            img[i][j][1] = 0
            img[i][j][2] = 255
        elif pred_image[i][j] == 2:   #白色
            img[i][j][0] = 255
            img[i][j][1] = 255
            img[i][j][2] = 255
        else:
            img[i][j][0] = 0           #黑色
            img[i][j][1] = 0
            img[i][j][2] = 0
cv2.imwrite('./gc.png',img)
pred_time_end = time.time()
print('预测耗费时间是:',pred_time_end - pred_time_start,'s')
print('预测结束!')
