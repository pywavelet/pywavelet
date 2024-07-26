import matplotlib.pyplot as plt
import os
import numpy as np

print("Now loading in analytical cov matrix")
Cov_Matrix_analytical_gap = np.load("Cov_Matrix_analytical_gap.npy")
print("Now loading in estimated covariance matrix")
Cov_Matrix_estm_gap = np.load("Cov_Matrix_estm_gap.npy")

fig,ax = plt.subplots(2,2, figsize = (16,8))
j = 0

ax[0,0].semilogy(np.abs(np.diag(Cov_Matrix_analytical_gap,j)),'*',label = 'analytical')
ax[0,0].semilogy(np.abs(np.diag(Cov_Matrix_estm_gap,j)),alpha = 0.7,label = 'estimated')
ax[0,0].set_title('Comparison of covariance matrices: Diagonal = {0}'.format(j))

ax[0,1].semilogy(np.abs(np.diag(Cov_Matrix_analytical_gap,j+10)),'*',label = 'analytical')
ax[0,1].semilogy(np.abs(np.diag(Cov_Matrix_estm_gap,j+10)),alpha = 0.7,label = 'estimated')
ax[0,1].set_title('Comparison of covariance matrices: Diagonal = {0}'.format(j+10))

ax[1,0].semilogy(np.abs(np.diag(Cov_Matrix_analytical_gap,j+50)),'*',label = 'analytical')
ax[1,0].semilogy(np.abs(np.diag(Cov_Matrix_estm_gap,j+50)),alpha = 0.7,label = 'estimated')
ax[1,0].set_title('Comparison of covariance matrices: Diagonal = {0}'.format(j+50))

ax[1,1].semilogy(np.abs(np.diag(Cov_Matrix_analytical_gap,j+100)),'*',label = 'analytical')
ax[1,1].semilogy(np.abs(np.diag(Cov_Matrix_estm_gap,j+100)),alpha = 0.7,label = 'estimated')
ax[1,1].set_title('Comparison of covariance matrices: Diagonal = {0}'.format(j+100))

for i in range(0,2):
    for j in range(0,2):
        ax[i,j].set_xlabel(r'index of matrix')
        ax[i,j].set_ylabel(r'magnitude')
        ax[i,j].legend()
        ax[i,j].grid()
plt.tight_layout()
plt.show()
