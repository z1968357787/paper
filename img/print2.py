from matplotlib import pyplot
import numpy as np
name_list=['ReLU','Sigmoid','Tanh','ELU','GELU']
two_layers_loss=[2.278910637, 11.99201774597168, 3.924001455307007, 6.647818565368652, 5.594772815704346]

three_layers_loss=[0.342410266, 12.673491477966309, 7.567915439605713, 9.369451522827148, 7.471736431121826]

four_layers_loss=[0.119474850, 3.5639591217041016, 0.8205013871192932, 1.490354299545288, 0.14731034636497498]

two_layers_test=[14.465725899, 25.583345413208008, 16.34505271911621, 15.185452461242676, 16.863826751708984]

three_layers_test=[11.210635185, 18.056488037109375, 13.284761428833008, 23.88529396057129, 13.514657020568848]

four_layers_test=[7.796854496, 13.199353218078613, 13.304327964782715, 8.635128021240234, 12.859424591064453]
#num_list3=np.absolute(np.array(num_list1)-np.array(num_list2))
#print(num_list3)

pyplot.bar(range(len(two_layers_loss)), two_layers_loss, fc='green', tick_label=name_list)
pyplot.title("2-layers-Linear")
pyplot.show()
pyplot.bar(range(len(three_layers_loss)), three_layers_loss, fc='yellow', tick_label=name_list)
pyplot.title("3-layers-Linear")
pyplot.show()
pyplot.bar(range(len(four_layers_loss)), four_layers_loss, fc='brown', tick_label=name_list)
pyplot.title("4-layers-Linear")
pyplot.show()

pyplot.bar(range(len(two_layers_test)), two_layers_test, tick_label=name_list)
pyplot.title("2-layers-Linear")
pyplot.show()
pyplot.bar(range(len(three_layers_test)), three_layers_test, fc='red', tick_label=name_list)
pyplot.title("3-layers-Linear")
pyplot.show()
pyplot.bar(range(len(four_layers_test)), four_layers_test, fc='blue', tick_label=name_list)
pyplot.title("4-layers-Linear")
pyplot.show()