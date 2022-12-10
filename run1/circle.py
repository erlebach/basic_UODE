import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.random import rand
from random import sample, shuffle
plt.close()
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from itertools import permutations
import matplotlib.pyplot as plt

# solution to the differential equation
t1 = 0.
t2 = 2.*np.arccos(-1)

# Equivalent to 
def circle (t, s):
    x = -s[1]
    y =  s[0]
    return x, y

# Equivalent to def circle above
#circle = lambda t, s: np.array([-s[1], s[0]])

F = circle

t0, tmax = 0, 6.28
npts = 200
t_eval = np.linspace(t0, tmax, npts)

# Computes accurate solution displayed at specified points. 

# Generate 20 initial condition curves: x0, y0 where x0 \in [0, 5], y0 \in [0, 5]
# This will generate 20 circles with different radii
# Also generate 20 array t_eval, with sorted random numbers between t0 and tmax. 
# Each list will contain Nt values. 

Nt = 200    # number of sampled points
Nruns = 1  # number of runs
x0max = 5
y0max = 5
dt = .01

x0 = x0max * rand(Nruns)
y0 = y0max * rand(Nruns)
x0 = np.asarray([x0max])
y0 = np.asarray([0.])
t_eval = np.linspace(0, tmax - dt, Nt)
t1 = np.tile(t_eval, (2,1)).transpose()
t1[:,1] = t1[:,0] + dt
t_eval = t1.reshape(-1)

sols = {}
y_list = []
for i in range(Nruns):
    # solution always evaluated at the same points
    sol = sols[i] = solve_ivp(F, [t0, tmax], [x0[i], y0[i]], t_eval=t_eval)
    plt.scatter(sol.t, sol.y[0], color='r')
    plt.scatter(sol.t, sol.y[1], color='b')
    plt.xlabel('t')
    plt.ylabel('sample points')
plt.show()
plt.close()


# select 100 triplets (t[2*i], x[2*i], x[2*i+1]) where x[2i] is the solution at t^n 
# and x[2i+1] is the solution at t^{n+1}
# Choose 100 even numbers between 0 and Nt-2
ix = Nruns * Nt

# Even numbers (the odd numbers are 'dt' away )
even = list(range(0, len(t_eval), 2))
Nsamples_per_run = 10
x0_pairs = []
x1_pairs = []
t_pairs = [] 
for i in range(Nruns):
    samples = np.array(sample(even, Nsamples_per_run)) 
    t_pairs.extend( list(zip(sol.t[samples], sol.t[samples+1])) )
    x0_pairs.extend( list(zip(sol.y[0][samples], sol.y[0][samples+1])) )
    x1_pairs.extend( list(zip(sol.y[1][samples], sol.y[1][samples+1])) )

#-----------------------------------------------------------------------------------
# I now have the training data on which to base the neural network: 
#   times:  (t_i^n, t_i^{n+1}), i=0, 1, Nruns*Nsamples_per_run
#   x0_pairs: (x[0]_i^n, x[0]_i^{n+1}), i=0, 1, Nruns*Nsamples_per_run
#   x1_pairs: (x[1]_i^n, x[1]_i^{n+1}), i=0, 1, Nruns*Nsamples_per_run

class NN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output
    
    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):
        super(NN, self).__init__()

        # input t into the network
        # There are two inputs and two outputs
        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 2)

        num_middle = num_hidden
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, t):
        #print(t.dtype)  # float
        #print(self.layer_in(t).dtype)
        out = self.act(self.layer_in(t))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        out = self.layer_out(out)
        return out

# Create DataSet and DataLoader 
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class myDataset(Dataset):
    def __init__(self, t_pairs, x0_pairs, x1_pairs):
        super(myDataset, self).__init__()
        self.t_pairs = torch.tensor(t_pairs, dtype=torch.float32)
        self.x0_pairs = torch.tensor(x0_pairs, dtype=torch.float32)
        self.x1_pairs = torch.tensor(x1_pairs, dtype=torch.float32)

    def __len__(self):
        return len(self.t_pairs)

    def __getitem__(self, i):
        return torch.tensor([*self.t_pairs[i], *self.x0_pairs[i], *self.x1_pairs[i]])


dataset = myDataset(t_pairs, x0_pairs, x1_pairs)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# Armed with the dataset and dataloader, I can implement the neural network
# For each batch, pass it through the NN, and output an approximation of d((x,y))/dt
# Apply Euler to obtain the solution at the next time step. Define MSE loss, and do back propagation
# The samples need not be in temporal order 

learning_rate = 1.e-3
loss_fn = torch.nn.MSELoss(reduction='mean')
model = NN(num_hidden=1, dim_hidden=30)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

nb_epochs = 400
losses = []
epoch_losses = []
for epoch in range(nb_epochs):
    # shuffle each epoch? 
    epoch_loss = 0
    count = 0
    for sample in dataloader:
        optimizer.zero_grad()  # correct place?
        s = sample[:, [2,4]]  # x[0]^n, x[1]^n
        dsdt_approx = model(s)
        snew = s + dt * dsdt_approx  # One step Euler
        s_exact = sample[:, [3,5]]
        s_approx = snew
        loss = loss_fn(snew, s_exact)
        epoch_loss += loss.item()
        loss.backward()
        losses.append(loss)
        optimizer.step()
        count += 1
    epoch_losses.append(epoch_loss / count)
    #print(epoch, loss.item())
print("epoch_losses: ", epoch_losses)

# With a trained NN, choose some initial condition and draw a trajectory (x, y)

sv = torch.tensor([1., 0.])

x_lst = []
y_lst = []
t_lst = []

dt1 = dt #* 0.5

for i in range(5000):
    t_lst.append(i*dt1)
    dydt = model(sv)
    sv = sv + dt1 * dydt
    x_lst.append(sv[0].item())
    y_lst.append(sv[1].item())

x_lst = np.asarray(x_lst)
y_lst = np.asarray(y_lst)
t_lst = np.asarray(t_lst)

plt.figure(figsize=(10, 6))
plt.plot(t_lst, x_lst, label='$x_0$')
plt.plot(t_lst, y_lst, label='$x_1$')
plt.xlabel('t')
plt.ylabel('$x_0$, $x_1$')
radius = np.sqrt(x_lst**2 + y_lst**2)
plt.plot(t_lst, radius, label='radius')
plt.legend()
plt.grid()
plt.show()
plt.close()

