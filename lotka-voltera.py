import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.random import rand
from random import sample, shuffle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from itertools import permutations
import matplotlib.pyplot as plt

# solution to the differential equation
t1 = 0.
t2 = 2.*np.arccos(-1)

# Parameters are global parameters. Not a good idea. 
alpha, beta, delta, gamma = 1.5, 3., 1., 1.

# Parameters from https://docs.juliahub.com/DiffEqFlux/BdO4p/1.13.0/examples/LV-Flux/
alpha, beta, delta, gamma = 2.2, 1.0, 2.0, 0.4

def lotka_voltera(t, s):
    # Parameters
    x = alpha*s[0] - beta*s[0]*s[1]  # dx/dt
    y = delta*s[0]*s[1] - gamma*s[1] # dy/dt
    return x, y

# I define F here and use F in the integrator
F = lotka_voltera

def euler(dt, nsteps, F, every=10):
    x0 = [1.,1.]
    x = x0
    x0_lst = []
    x1_lst = []
    t_lst = []
    for i in range(nsteps):
        f = F(0., x)
        x[0] = x[0] + dt * f[0]
        x[1] = x[1] + dt * f[1]
        if i % every == 0:
            print("x: ", x)
            t_lst.append(dt*i)
            x0_lst.append(x[0])
            x1_lst.append(x[1])
    return t_lst, x0_lst, x1_lst

"""
t_lst, x0_lst, x1_lst = euler(0.0001, 800000, F, every=1000)
plt.plot(t_lst, x0_lst)
plt.plot(t_lst, x1_lst)
plt.show()
plt.close()
quit()
"""

#npts = 200
#t_eval = np.linspace(t0, tmax, npts)

# Computes accurate solution displayed at specified points. 

# Generate 20 initial condition curves: x0, y0 where x0 \in [0, 5], y0 \in [0, 5]
# This will generate 20 circles with different radii
# Also generate 20 array t_eval, with sorted random numbers between t0 and tmax. 
# Each list will contain Nt values. 

# ISSUE: Lotka-Voltera equations are stiff. I need dt=0.0001 when running with Euler method. 
# Therefore, I need a different approach. 

Nt = 120    # number of sampled points
Nruns = 1  # number of runs
dt = .1
t0, tmax = 0, 10.+dt
batch_size = 10

# In this approach, the points at which I evaluate the exact solution is decoupled from 
# the time step dt required for the network training. 

#x0 = rand(Nruns)
#y0 = rand(Nruns)
x0 = np.asarray([1.])
y0 = np.asarray([1.])
t_eval = np.linspace(0., tmax, Nt)  # Number of sampled times
#t1 = np.tile(t_eval, (2,1)).transpose()
#t1[:,1] = t1[:,0] + dt
#t_eval = t1.reshape(-1)

# Calculate solution at t_eval points, with an adaptive integrator
sols = {}
y_list = []
for i in range(Nruns):
    # solution always evaluated at the same points
    sol = sols[i] = solve_ivp(F, [t0, tmax], [x0[i], y0[i]], t_eval=t_eval)

"""  (max of about 7)
# Plot solution
for i in range(Nruns):
    sol = sols[i]
    plt.scatter(sol.t, sol.y[0], color='r')
    plt.scatter(sol.t, sol.y[1], color='b')
    plt.xlabel('t')
    plt.ylabel('sample points')
plt.show()
plt.close()
"""

# Select training data: Nsamples_per run points at random. 
# Even numbers (the odd numbers are 'dt' away )

# subtrace 1 from Nsamples_per_run since I am constructing pairs, whose
# second elements will be the label (i.e., the exact solution)
Nsamples_per_run = 100
x0_pairs = []
x1_pairs = []
t_pairs = [] 

# Take samples from all the runs. (in this case, only a single run if Nruns==1)
indexes = list(range(0, len(t_eval)-1))
print(len(indexes))
for i in range(Nruns):
    samples = np.array(sample(indexes, Nsamples_per_run)) 
    t_pairs.extend(  list(zip(sol.t[samples],    sol.t[samples+1]))    )
    x0_pairs.extend( list(zip(sol.y[0][samples], sol.y[0][samples+1])) )
    x1_pairs.extend( list(zip(sol.y[1][samples], sol.y[1][samples+1])) )

#quit()

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
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(normalized_shape=[30]) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, t):
        #print(t.dtype)  # float
        #print(self.layer_in(t).dtype)
        #print(t.shape)
        out = self.act(self.layer_in(t))
        for layer, ln in zip(self.middle_layers, self.layer_norms):
            #out = ln(out)
            out = self.act(layer(out))
        out = self.layer_out(out)   # No activation on the last layer
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
        # The 6 elements: t^n, t^{n+1}, y[0]^n, y[0]^{n+1}, y[1]^n, y[1]^{n+1}
        #return torch.tensor([*self.t_pairs[i], *self.x0_pairs[i], *self.x1_pairs[i]])
        # Two triplets (t,x1,x2)^n, (t,x1,x2)^{n+1}
        #print(self.t_pairs[i][0].item())
        s = self
        return torch.tensor([s.t_pairs[i][0].item(), s.x0_pairs[i][0].item(), s.x1_pairs[i][0].item(), 
                    s.t_pairs[i][1].item(), s.x0_pairs[i][1].item(), s.x1_pairs[i][1].item()]).reshape(2,3)


dataset = myDataset(t_pairs, x0_pairs, x1_pairs)
# drop_last: ignore the last batch, if not of the proper size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Armed with the dataset and dataloader, I can implement the neural network
# For each batch, pass it through the NN, and output an approximation of d((x,y))/dt
# Apply Euler to obtain the solution at the next time step. Define MSE loss, and do back propagation
# The samples need not be in temporal order 

learning_rate = 1.e-1
loss_fn = torch.nn.MSELoss(reduction='mean')
# poor results with ReLU
model = NN(num_hidden=2, dim_hidden=5, act=nn.Tanh())
# loss going down to half of initial value
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

nb_epochs = 200
losses = []
epoch_losses = []

for epoch in range(nb_epochs):
    # shuffle each epoch? 
    epoch_loss = 0
    count = 0
    for sample in dataloader:  
        # Sample: 2D array: (batch_size, 6)
        # The 6 elements: t^n, t^{n+1}, y[0]^n, y[0]^{n}, y[1]^n, y[1]^{n}
        optimizer.zero_grad()  # correct place?
        # I have a batch of 3
        s = sample[:, 0, 1:]  # x[0]^n, x[1]^n  (first dimension is batch_size)
        dsdt_approx = model(s)

        # Euler method
        # s and snew do not have requires_grad on
        linear = [ alpha*s[:,0], -gamma*s[:,1] ]
        linear = torch.concat(linear).reshape([2, batch_size]).transpose(1,0)
        snew = s + dt * (linear + dsdt_approx)

        # linear and s do not have requires_grad turned on. snew does, because of dsdt_approx. 
        # but since s and linear do not depend on the parameters, that should not be a problem.

        #  batch_size, 2, 3
        #  2, 3: t^n, x[0]^n, x[1]^n, t^{n+1}, x[0]^{n+1], x[1]^{n+1}
        #print(sample.shape)
        s_exact = sample[:, 1, 1:]  # x[0]^{n+1}, x[1]^{n+1}

        loss = loss_fn(snew, s_exact)
        epoch_loss += loss.item()
        loss.backward()
        losses.append(loss)
        optimizer.step()
        count += 1

    # LOSSES ARE NOT DECREASING!
    epoch_losses.append(epoch_loss / count)

    if epoch % 10 == 0:
        print("loss: ", epoch_loss / count)

# THE LOSSES MUST DECREASE
print("epoch_losses: ", epoch_losses)

# With a trained NN, choose some initial condition and draw a trajectory (x, y)

s_initial = torch.tensor([1., 1.])

x_lst = []
y_lst = []
t_lst = []

dt1 = dt 

optimizer.zero_grad()


"""
# Print out output of neural network without an associated ODE. Use the a and y from the exact solution
# The Neural network should be modeling alpha*x*y. To be more specific: NN[0] / NN[1] should be constant. 
#print(sol.y)
with torch.no_grad():
    for i, t in enumerate(sol.t):
        sv = torch.tensor([sol.y[0][i], sol.y[1][i]], dtype=torch.float32)
        t_lst.append(i * dt)
        #print("t, i*dt: ", t, i*dt)
        dydt_NN = model(sv)
        x_lst.append(dydt_NN[0].item())
        y_lst.append(dydt_NN[1].item())
quit()

print(sol.t)
plt.plot(sol.t, x_lst)
plt.plot(sol.t, y_lst)
plt.title("Output from NN as a function of t")
plt.show()
plt.close()
quit()
"""

t_lst = []
x_lst = []
y_lst = []

# Solve the original ODE with the NN instead of unknown terms
# Output exact solution at the same points where computed 
nsteps = 600
t_eval = np.linspace(0., 600*dt1, nsteps)
i = 0 # 0th run
sol = sols[i] = solve_ivp(F, [t0, tmax], [x0[i], y0[i]], t_eval=t_eval)

with torch.no_grad():
    sv = s_initial
    for i in range(nsteps):
        t_lst.append(i * dt1)
        #print("t vs t: ", i*dt1, sol.t[i])
        dydt_NN = model(sv)
        linear = torch.tensor([ alpha*sv[0].item(), -gamma*sv[1].item() ])
        sv = sv + dt1 * (linear + dydt_NN)
        x_lst.append(sv[0].item())
        y_lst.append(sv[1].item())

x_lst = np.asarray(x_lst)
y_lst = np.asarray(y_lst)
t_lst = np.asarray(t_lst)
#print("t_lst: ", t_lst[0:500])
#print("x_lst: ", x_lst[0:500])
#print("y_lst: ", y_lst[0:500])

plt.close()
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='$x_{0, exact}$', lw=3, color='red')
plt.plot(sol.t, sol.y[1], label='$x_{1, exact}$', lw=3, color='red')
plt.plot(t_lst, x_lst, label='$x_0$', lw=2, color='green')
plt.plot(t_lst, y_lst, label='$x_1$', lw=2, color='blue')
plt.xlabel('t')
plt.ylabel('$x_0$, $x_1$')
#radius = np.sqrt(x_lst**2 + y_lst**2)
#plt.plot(t_lst, radius, label='radius')
plt.legend()
plt.grid()
plt.show()
plt.close()

