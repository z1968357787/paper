from matplotlib import pyplot
import numpy as np
name_list=['2-times','4-times','6-times','8-times','10-times']
RNN_loss=[0.7251885533332825,0.6658056378364563,1.113870502,1.1534407138824463,0.9984080195426941]

LSTM_loss=[0.8961564898490906,0.398283451795578,0.582557619,0.844742476940155,0.7428650259971619]

GRU_loss=[0.64846271276474,0.5013042688369751,0.782070577,0.5582277774810791,0.4813990890979767]

RNN_test=[28.838518142700195,23.0535831451416,16.735458374,17.30996322631836,27.965774536]

LSTM_test=[24.603076934814453,19.69328498840332,12.859604836,18.01136016845703,22.904857635498047]

GRU_test=[21.540735244750977,18.063831329345703,14.584783554,20.18978500366211,23.470348358154297]
#num_list3=np.absolute(np.array(num_list1)-np.array(num_list2))
#print(num_list3)

pyplot.bar(range(len(RNN_loss)),RNN_loss,fc='green',tick_label=name_list)
pyplot.title("min_train_loss")
pyplot.show()
pyplot.bar(range(len(LSTM_loss)),LSTM_loss,fc='yellow',tick_label=name_list)
pyplot.title("test_loss")
pyplot.show()
pyplot.bar(range(len(GRU_loss)),GRU_loss,fc='brown',tick_label=name_list)
pyplot.title("|test_loss-min_train_loss|")
pyplot.show()

pyplot.bar(range(len(RNN_test)),RNN_test,tick_label=name_list)
pyplot.title("min_train_loss")
pyplot.show()
pyplot.bar(range(len(LSTM_test)),LSTM_test,fc='red',tick_label=name_list)
pyplot.title("test_loss")
pyplot.show()
pyplot.bar(range(len(GRU_test)),GRU_test,fc='blue',tick_label=name_list)
pyplot.title("|test_loss-min_train_loss|")
pyplot.show()