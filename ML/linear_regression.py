import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# plt.style.use('./deeplearning.mplstyle')  # Commented out - file doesn't exist
plt.style.use('default')  # Using default matplotlib style instead
np.set_printoptions(precision=2)

# training set 
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# print training set
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

# init w & b
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(w_init)
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

def predict(x, w, b):
  p = np.dot(x, w) + b
  return p

def compute_cost(x, y, w, b):
    m = x.shape[0]  # number of training examples
    cost = 0
    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        cost += (f_wb - y[i]) ** 2
    cost /= (2 * m)
    return cost

def compute_gradient(X, y, w, b):
  """
  Computes the gradient for linear regression 
  Args:
    X (ndarray (m,n)): Data, m examples with n features
    y (ndarray (m,)) : target values
    w (ndarray (n,)) : model parameters  
    b (scalar)       : model parameter
    
  Returns:
    dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
    dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
  """

  m = X.shape[0]
  n = X.shape[1]

  dj_dw = np.zeros(n)
  dj_db = 0
  for i in range(m):
    error = np.dot(X[i], w) + b - y[i]
    for j in range(n):
      dj_dw[j] += error * X[i, j]
    
    dj_db += error

  return dj_db / m, dj_dw / m


x_vec = X_train[0]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
print(f"target = {y_train[0]}")

print(np.dot(X_train, w_init) + b_init)

# Compute and display cost using our pre-chosen optimal parameters.
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}') 

#Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')