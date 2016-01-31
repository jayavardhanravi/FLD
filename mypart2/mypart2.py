import numpy as lib_npy
import matplotlib.pyplot as lib_graph
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy.linalg import inv

lib_npy.random.seed(0)

# Creating values of mean of both classes and accuracy at each point in the range
mean_X_start = 0
mean_X_end = 3
mean_Y_start = 0
mean_Y_end = 3

No_points = 50
var_factor_X = (float)((mean_X_end-mean_X_start)/No_points); 
var_factor_Y = (float)((mean_Y_end-mean_Y_start)/No_points); 
iter_points = No_points+1

mean_X_val = []
mean_Y_val = []

# Calculating the means
for var1 in range(iter_points):
	for var2 in range(iter_points):
		mean_X_val.append(var1*var_factor_X)
		mean_Y_val.append(var2*var_factor_Y)
		
# Calculating the accuracy
##########################################################################################

mean_accuracy_val = []
#fVar = open('Generated_dataset_part_2.dat','w') 

for var1 in range(iter_points):
	for var2 in range(iter_points):
		# Defining the given input data
		mean_X = lib_npy.array([-3+(var1*var_factor_X),4+(var1*var_factor_X)])
		cov_X = [[1,0],[0,1]]
		N_X = 20
		mean_Y = lib_npy.array([3+(var2*var_factor_Y),-2+(var2*var_factor_Y)])
		cov_Y = [[2,0],[0,2]]
		N_Y = 20

		# Generating the dataset and printing to file
		gen_data_X = lib_npy.random.multivariate_normal(mean_X,cov_X,20).T
		#print gen_data_X
		#fVar = open('Generated_dataset.dat','w')
		#for i in range(0,20):
			#fVar.write("%f %f 0\n" %(gen_data_X[0][i],gen_data_X[1][i]))
		#fVar.close()
		gen_data_Y = lib_npy.random.multivariate_normal(mean_Y,cov_Y,20).T
		#print gen_data_Y
		#with open('Generated_dataset.dat', 'a') as fVar1:
			#for i in range(0,20):
				#fVar1.write("%f %f 1\n" %(gen_data_Y[0][i],gen_data_Y[1][i]))
		#fVar1.close()

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
		mean_accuracy_val.append(accuracy_cal)
		
		# Printing data to file
		#fVar.write("%f %f %f\n" %((var1*var_factor_X),(var2*var_factor_Y),accuracy_cal))

#fVar.close()

##################################################################################################

# Plotting the 3D graph
# Reference : http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#surface-plots

my3D_plot = lib_graph.figure()
my3D_proj = my3D_plot.gca(projection='3d')
mean_X_val = lib_npy.arange(0, 3, 0.006)
mean_Y_val = lib_npy.arange(0, 3, 0.006)
mean_X_val, mean_Y_val = lib_npy.meshgrid(mean_X_val, mean_Y_val)
my3D_angle = lib_npy.sqrt(mean_X_val**2 + mean_Y_val**2)
mean_accuracy_val = lib_npy.sin(my3D_angle)
my3D_surf = my3D_proj.plot_surface(mean_X_val, mean_Y_val, mean_accuracy_val, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
my3D_proj.set_zlim(-1.5, 1.5)
my3D_proj.zaxis.set_major_locator(LinearLocator(10))
my3D_proj.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
my3D_plot.colorbar(my3D_surf, shrink=0.5, aspect=5)
my3D_proj.set_xlabel('Mean for class 0')
my3D_proj.set_ylabel('Mean for class 1')
my3D_proj.set_zlabel('Accuracy')
lib_graph.title('Graph of mean for class 0, mean for class 1 and accuracy')
lib_graph.show()

