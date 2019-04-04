# Satellite-cloud-image-classification-based-on-multi-dimensional-multi-grained-cascade-forest
关于数据集:
   本代码所采用的数据集均采自中国坏境卫星HJ1A/1B．
   数据集中的样本的维度是28x28x4．
   样本分为4类，分别是云区域，雪区域，无云无雪区域，云雪混合区域．
   类别0表示云区域，用红色表示．
   类别1表示雪区域，用蓝色表示．
   类别2表示云雪混合区域，用白色表示．
   类别3表示无云无雪区域，用黑色表示．
关于Python版本:
   本代码使用Python3.6
   使用sklearn
   使用opencv
   使用numpy
   使用random
关于模型:
   使用的模型是多维多粒度级联森林模型，是一种不同于深度神经网络的深度学习模型．该模型是周志华教授提出的．模型具有较好的泛化效果．
