# import
# torch / main, has tensor, math functions
# torch.autograd / for derivative functions
# torch.nn / neural network, data layers
# torch.multiprocessing / multi processing
# torch.optim / give algorithm for parameter optimization
# torch.utils / data manipulation
# torch.onnx / when diff frameworks share a model


# tensor / representation of data, think of it as (blocks, matrix)
# ex, scalar rank 0 shape() / vector rank 1 shape(3) / matrix rank 2 shape(3) / matrix rank 3 shape(3,3,3)

import torch
# TENSOR DECLARE
"""
x = torch.empty(4,2)
print(x)
x = x.new_ones(3,3,3, dtype = torch.double)
print(x)
print("x size is: ", x.size())
#print(x.item()) # this dont wokr cuz its not scalar
y = torch.rand(3,3,3)
print(y)
z = x + y
print("this is sum of a+b which is z: ", z)
print("this is dimension of z: ", z.dim())
"""

import math
# TENSOR MATH
"""
x = torch.tensor([[1,2], [3,4], [5,6]])
print(x.size())
y = x + 1 # we add one to every elements in x tensor
print(y)
print(torch.min(x))
print(torch.max(x))
#print(torch.mean(x))
#print(torch.std(x))
print(torch.prod(x))
z = torch.rand(3,2)
l = torch.rand(2,1)
print(torch.add(x,z)) #same as just x + z
print(torch.mm(z, l)) # this is dot product
"""

# TENSOR MANIPULATION
"""
x = torch.Tensor([[1,2], [3,4]])
print(x[1,0])
print(x[0])
print(x[:, 1])
y = torch.rand(3, 6)
print(y)
print(y.view(18)) # make 3x6 flat
print(y.view(2, -1)) # make 2 rows, and the rest, you take care of it
z = torch.rand(1,3,4)
print(z)
z_sq = z.squeeze()
print(z_sq.shape) # only squeeze when you have 1 dim
z_unsq = z.unsqueeze(dim = 0) # add dimension in dim index 0
print(z_unsq.size())
z_unsq1 = z.unsqueeze(dim = 2) # add dimension in dim index 0
print(z_unsq1.size())
"""

import torch.autograd
# TENSOR AUTOGRAD (derivative)
"""
x = torch.rand(4, 3)
x = x * 2
print(x)
print(x.requires_grad) #requires grad track the history
x.requires_grad_(True) # now we can keep track the history
x = x + 2
print(x)
print(x.grad_fn) # this will show that we 'add' 
"""
"""
x = torch.ones(3,3,3, requires_grad= True)
print(x)
y = x + 2
print(y)
z = y * y
meann = z.mean()
print(z, meann)
meann.backward() # calculate how we can change 'weight' to prevent less loss,
# backward go with 'loss' var, to calculate how to change 'weight' to prevent less loss.
# and then the calculation will be stored in 'weight.grad'
# then we change 'weight', optimizer will see the 'grad' value and change 'weight' accordignly
"""

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
# LOADING DATA 





