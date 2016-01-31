import numpy as lib_npy
import matplotlib.pyplot as lib_graph
from numpy.linalg import inv

lib_npy.random.seed(0)

# Defining the given input data
mean_X = lib_npy.array([-3,4])
cov_X = [[1,0],[0,1]]
N_X = 20
mean_Y = lib_npy.array([3,-2])
cov_Y = [[2,0],[0,2]]
N_Y = 20

# Generating the dataset and printing to file
gen_data_X = lib_npy.random.multivariate_normal(mean_X,cov_X,20).T
#print gen_data_X
fVar = open('Generated_dataset_part_1.dat','w')
for i in range(0,20):
	fVar.write("%f %f 0\n" %(gen_data_X[0][i],gen_data_X[1][i]))
fVar.close()
gen_data_Y = lib_npy.random.multivariate_normal(mean_Y,cov_Y,20).T
#print gen_data_Y
with open('Generated_dataset_part_1.dat', 'a') as fVar1:
	for i in range(0,20):
		fVar1.write("%f %f 1\n" %(gen_data_Y[0][i],gen_data_Y[1][i]))
fVar1.close()

# Calculating the means
mean_cal_X0 = lib_npy.mean(gen_data_X[0])
mean_cal_X1 = lib_npy.mean(gen_data_X[1])
#print mean_cal_X0
#print mean_cal_X1
mean_cal_Y0 = lib_npy.mean(gen_data_Y[0])
mean_cal_Y1 = lib_npy.mean(gen_data_Y[1])
#print mean_cal_Y0
#print mean_cal_Y1
mean_f = [[mean_cal_X0,mean_cal_X1],[mean_cal_Y0,mean_cal_Y1]]
mean_arr_final=[[],[]]	
mean_arr_final[0] = lib_npy.zeros((2,20))
mean_arr_final[1] = lib_npy.zeros((2,20))
mean_arr_final[0][0].fill(mean_cal_X0)
mean_arr_final[0][1].fill(mean_cal_X1)
mean_arr_final[1][0].fill(mean_cal_Y0)
mean_arr_final[1][1].fill(mean_cal_Y1)
#print mean_arr_final[0]
#print mean_arr_final[1]

# Calculating the within class scatter matrix
scatt_within = lib_npy.zeros((2,2))
scatt_within = scatt_within + (gen_data_X[0]-mean_arr_final[0]).dot((gen_data_X[0]-mean_arr_final[0]).T) + (gen_data_Y[1]-mean_arr_final[1]).dot((gen_data_Y[1]-mean_arr_final[1]).T)
#print scatt_within

# Calculating the between class scatter matrix
scatt_between = lib_npy.zeros((2,2))
scatt_between = (mean_arr_final[1]-mean_arr_final[0]).dot((mean_arr_final[1]-mean_arr_final[0]).T)
#print scatt_between

# Calculating the W
W_cal = lib_npy.zeros((2,2))
mean_diff = [[(mean_f[1][0]-mean_f[0][0])],[(mean_f[1][1]-mean_f[0][1])]]
W_cal = inv(scatt_within).dot(mean_diff)
#print W_cal

# Calculating the accuracy
count_class_0 = 0
count_class_1 = 0
for var in range(20):
	if((gen_data_X[0][var]*W_cal[0]+gen_data_X[1][var]*W_cal[1])<0):
		count_class_0 = (count_class_0)+1
	if((gen_data_Y[0][var]*W_cal[0]+gen_data_Y[1][var]*W_cal[1])>=0):
		count_class_1 = (count_class_1)+1
		
#print count_class_0
#print count_class_1
accuracy_cal = float((count_class_0+count_class_1)/(20+20))*100
print "The accuracy of seperating the classes is %.2f%%\n" %(accuracy_cal)

# Plotting the two classes and decision boundry
lib_graph.plot(gen_data_X[0],gen_data_X[1],'ro',label="X")
lib_graph.plot(gen_data_Y[0],gen_data_Y[1],'bo',label="Y")
lib_graph.plot([-5,5],[5*W_cal[0]/W_cal[1],-5*W_cal[0]/W_cal[1]],color='red',label="Boundry")
lib_graph.title('Graph of the X,Y and decision boundry')
lib_graph.legend()
lib_graph.show()

