

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C











def coreAlg(X,y):


	X=np.atleast_2d(np.array(X)).T
	y=np.atleast_2d(np.array(y)).T


	#plt.subplots(6,6,figsize=(30,30))
	fig, axes = plt.subplots(nrows=4, ncols=4,figsize=(12,12))


	count = 0
	for i in [0.1, 0.5, 1, 1.5]:
		for j in [0.1, 0.5, 1, 1.5]:
			count=count+1
			kernel = C(i) * RBF(j)




			gp = GaussianProcessRegressor(kernel=kernel,alpha=0.5*0.5,optimizer = None)
			#gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=100)

			gp.fit(X, y)







			n = 10
			x = np.atleast_2d(np.array(np.linspace(-4, 4, n))).T




			y_pred, sigma = gp.predict(x, return_std=True)
			plt.subplot(4,4,count)
			plt.plot(X, y, 'r.', markersize=8)
			plt.plot(x, y_pred, 'b-')
			plt.axis([-4, 4, -3, 3])
			plt.xticks(fontsize=6)

			plt.fill_between(x.ravel(),y_pred.ravel()-1.96*sigma,y_pred.ravel()+1.96*sigma,alpha=0.5,color="#dddddd")
			plt.title("lambda = " + str(i) + ", l = " +str(j),fontsize=8)
		

			#Thet=gp.log_marginal_likelihood()

			#print(Thet)

		    





			      

	#plt.legend(loc='upper left')
	plt.tight_layout()
	#plt.show()
	plt.savefig('q4_2.pdf')


X = [-2.26, -1.31, -0.43, 0.32, 0.34, 0.54, 0.86, 1.83, 2.77, 3.58]
y = [1.03, 0.70, -0.68, -1.36, -1.74, -1.01, 0.24, 1.55, 1.68, 1.53]

coreAlg(X,y)





