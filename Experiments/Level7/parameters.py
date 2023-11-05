# Reinitialization parameters
alpha_dif = 8*1e-4 # diffusion parameter
dt = 1.5*1e-4
Ninit = 50

# TV Regularization parameter
beta = 5e-9
# TV Smoothness parameter in lagged diffusion
delta = 1e-3

# Smoothness parameter for delta approximation and sign function in reinitialization
eps = 1e-3
# Step size
alpha = 0.05
# Number of iterations
Miter = 200

# Backtracking line search
#tau = 0.5
#c = 0.025

# Starting guess
r = 0.23/4
k = np.linalg.norm(g, axis=1)
phi0 = k-r
q1 = 0.8
q2 = 1e-2