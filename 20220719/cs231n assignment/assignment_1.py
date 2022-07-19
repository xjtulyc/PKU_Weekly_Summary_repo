# 从当前路径下的KNN.py载入KNN函数
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset.cifar_dataset import testloader
from knn.knn import KNN

start_time = time.clock()

num_train = 10000
num_test = 5000

# K的取值
k_choice = np.array([5, 10, 15, 20, 40, 100])
# k_choice=[10]
k_accuracy = np.zeros_like(k_choice)
for i, k_c in enumerate(k_choice):
    for idx, (image, label) in enumerate(tqdm(testloader)):
        predict_label = KNN(image, k_c)
        if predict_label == label:
            k_accuracy[i] += 1
            pass
    print('k: {}; acc: {}'.format(k_c, k_accuracy[i] / num_test))

k_accuracy /= num_test

# 将结果可视化
plt.figure(figsize=(10, 6))
plt.plot(k_choice, k_accuracy, 'r-')
plt.plot(k_choice, k_accuracy, 'go')
plt.xlabel('k')
plt.ylabel('accuracy(%)')
plt.title('k choice')
plt.show()

# 记录最大精度以及对应K值
max_id = np.argmax(k_accuracy)
max_acc = k_accuracy[max_id]
max_acc_k = k_choice[max_id]

print('\n\nmax_acc=%.2f%%,max_acc_k=%d' % (max_acc, max_acc_k))

end_time = time.clock()

print('\n\n运行时间：%ss' % (str(end_time - start_time)))
