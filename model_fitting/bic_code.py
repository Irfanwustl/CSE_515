import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, ExpSineSquared,\
                                             Matern, PairwiseKernel, RBF, RationalQuadratic, WhiteKernel,\
                                             CompoundKernel, Product,Exponentiation, Hyperparameter, Kernel, Sum

kernels_norm_bic = [1.0, DotProduct(sigma_0=1.0,),ExpSineSquared(length_scale=1.0, periodicity=1.0), Matern(length_scale=1.0),\
           PairwiseKernel(gamma=1),RBF(length_scale=1.0),RationalQuadratic(length_scale=1, alpha=1),WhiteKernel(noise_level=1.0)]

test_kernels_norm_bic = []
best_kernels_norm_bic = []

for i in range (len(kernels_norm_bic)):
  test_kernels_norm_bic.append(kernels_norm_bic[i])
  for j in range (len(kernels_norm_bic)):
    if (i != j and kernels_norm_bic[j]+kernels_norm_bic[i] not in test_kernels_norm_bic and kernels_norm_bic[j]*kernels_norm_bic[i] not in test_kernels_norm_bic):
      test_kernels_norm_bic.append(kernels_norm_bic[i]+kernels_norm_bic[j])
      test_kernels_norm_bic.append(kernels_norm_bic[i]*kernels_norm_bic[j])

lowest_norm_bic = float("inf")
index_norm_bic = 0
print("(Norm) Kernel list:")
for i, kernel in enumerate(test_kernels_norm_bic):
  try:
    gpr_norm_bic = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr_norm_bic.fit(xTrain_norm, yTrain_norm)
    mu_s_norm_bic, std_s_norm_bic = gpr_norm_bic.predict(xTest_norm, return_std=True)

    bic_norm = 3.0 * np.log(len(xTrain_norm)) - 2 * gpr_norm_bic.log_marginal_likelihood()
    print (i, ": ",kernel, "= ", bic_norm)
    if (bic_norm < lowest_norm_bic):
        lowest_norm_bic = bic_norm
        best_kernel_norm_bic = kernel
        index_norm_bic = i
    if (bic_norm < 400):
        best_kernels_norm_bic.append((i, kernel,bic_norm))

  except:
    print (i, ": ",kernel, "= Error")

if(best_kernels_norm_bic):
    print("(Norm) Best kernels as follows:")
    for i, kernel in enumerate(best_kernels_norm_bic):
        print(kernel[0], ": ", kernel[1], "= ", kernel[2])

    #print (best_kernels)
print("(Norm) Lowest Kernel: %d, %s, %s " %(index_norm_bic, best_kernel_norm_bic, lowest_norm_bic))
