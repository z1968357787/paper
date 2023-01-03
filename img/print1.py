from matplotlib import pyplot
import numpy as np
name_list=['2-Layers','3-Layers','4-Layers','RNN','LSTM','GRU']

train_loss=[2.2789106369018555, 0.34241026639938354, 0.1194748505949974, 1.1138705015182495, 0.5825576186180115, 0.7820705771446228]
test_loss=[14.465725898742676, 11.2106351852417, 7.796854496002197, 16.735458374023438, 12.859604835510254, 14.58478355407715]

test_train=np.absolute(np.array(test_loss) - np.array(train_loss))
#print(num_list3)

pyplot.bar(range(len(train_loss)), train_loss, tick_label=name_list)
pyplot.title("min_train_loss")
pyplot.show()
pyplot.bar(range(len(test_loss)), test_loss, fc='red', tick_label=name_list)
pyplot.title("test_loss")
pyplot.show()
pyplot.bar(range(len(test_train)), test_train, fc='blue', tick_label=name_list)
pyplot.title("|test_loss-min_train_loss|")
pyplot.show()