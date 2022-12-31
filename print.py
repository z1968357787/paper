from matplotlib import pyplot
import numpy as np
name_list=['2-Layers','3-Layers','4-Layers','RNN','LSTM','GRU']
num_list1=[13.792022705078125,6.5560173988342285,6.65120267868042,0.8782229423522949,0.5606899261474609,0.6172841787338257]

num_list2=[11.727431297302246,7.127485275268555,10.827729225158691,16.22555923461914,12.429522514343262,14.287299156188965]

num_list3=np.absolute(np.array(num_list1)-np.array(num_list2))
print(num_list3)

pyplot.bar(range(len(num_list2)),num_list2,tick_label=name_list)
pyplot.show()
pyplot.bar(range(len(num_list1)),num_list1,fc='red',tick_label=name_list)
pyplot.show()
pyplot.bar(range(len(num_list3)),num_list3,fc='blue',tick_label=name_list)
pyplot.show()