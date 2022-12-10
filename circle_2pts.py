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

# The training data is two points x^{n-1}, x^n, separated by dt. 
# We will integrate the equations with dt, and choose Nt random points
# per solution, and choose three points (the third point is the label for 
# the Euler algorithm. 

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

# Time interval
t0, tmax = 0, 6.28

# Generate 20 initial condition curves: x0, y0 where x0 \in [0, 5], y0 \in [0, 5]
# This will generate 20 circles with different radii
# Also generate 20 array t_eval, with sorted random numbers between t0 and tmax. 
# Each list will contain Nt values. 

Nruns = 150   # number of runs
x0max = 1
y0max = 1
dt = .02

# Alternative approach: 
# Generate a collection of points (x[n-1], x[n], x[n+1] (for initial conditions)
x0 = x0max * rand(Nruns)
y0 = y0max * rand(Nruns)

# Only work with a single run and validate on the same run (easier problem)
#x0 = x0max * np.ones([Nruns])
#y0 = y0max * np.ones([Nruns])
t0 = t1 + (t2-t1) * rand(Nruns)  # starting times, random between [0, 6.28]
#x0 = np.asarray([x0max])


sols = {}
y_list = []
for i in range(Nruns):
    # solution always evaluated at the same points
    t_eval = np.linspace(t0[i], t0[i]+2*dt, 3)
    # x[0][i] and y0[i] are always 1
    # What is the second argument of solve_ivp?  Min/max t for simulation
    # t_eval are three time points: t0[i], t0[i]+dt, t0[i]+2*dt
    # Initial Conditions (I.C.): x0[i], y[0][i] for run i
    sol = sols[i] = solve_ivp(F, [t0[i], t0[i]+2*dt], [x0[i], y0[i]], t_eval=t_eval)
    print(f"({i}) t_eval: ", t_eval)
    print(f"({i}) t[i]: ", t0[i])
    print(f"({i}) x0[i]: ", x0[i])
    print(f"({i}) y0[i]: ", y0[i])
    print()
    #print(sol.t)
    plt.plot(sol.t, sol.y[0], color='r', lw=1)
    plt.plot(sol.t, sol.y[1], color='b', lw=1)
    #plt.scatter(sol.t, sol.y[0], color='r')
    #plt.scatter(sol.t, sol.y[1], color='b')
    plt.xlabel('t')
    #plt.ylabel('sample points')
plt.show()
plt.close()
#quit()

# select 100 triplets (t[2*i], x[2*i], x[2*i+1]) where x[2i] is the solution at t^n 
# and x[2i+1] is the solution at t^{n+1}

# Even numbers (the odd numbers are 'dt' away )
even = list(range(0, len(t_eval), 2))
Nsamples_per_run = 10
x0_seq3 = []
x1_seq3 = []
t_seq3 = [] 
for i in range(Nruns):
    sol = sols[i]
    t_seq3.append( list(sol.t) )
    x0_seq3.append( list(sol.y[0]) )
    x1_seq3.append( list(sol.y[1]) )

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
        # The inputs are x^{n-1}, x^{n} where x \in R^2
        self.layer_in = nn.Linear(4, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 2)

        num_middle = num_hidden
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, t):
        out = self.act(self.layer_in(t))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        out = self.layer_out(out)
        return out

# Create DataSet and DataLoader 
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class myDataset(Dataset):
    def __init__(self, t_seq, x0_seq, x1_seq):
        super(myDataset, self).__init__()
        self.t_seq = torch.tensor(t_seq, dtype=torch.float32)
        self.x0_seq = torch.tensor(x0_seq, dtype=torch.float32)
        self.x1_seq = torch.tensor(x1_seq, dtype=torch.float32)

    def __len__(self):
        return len(self.t_seq // 3)

    def __getitem__(self, i):
        # return 3 sequences of size 3
        return torch.tensor([*self.t_seq[i], *self.x0_seq[i], *self.x1_seq[i]])

dataset = myDataset(t_seq3, x0_seq3, x1_seq3)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

# Armed with the dataset and dataloader, I can implement the neural network
# For each batch, pass it through the NN, and output an approximation of d((x,y))/dt
# Apply Euler to obtain the solution at the next time step. Define MSE loss, and do back propagation
# The samples need not be in temporal order 

learning_rate = 1.e-2
loss_fn = torch.nn.MSELoss(reduction='mean')
model = NN(num_hidden=2, dim_hidden=30, act=nn.ReLU())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Loss function is hardly decreasing. WHY? 
nb_epochs = 400
losses = []
epoch_losses = []
for epoch in range(nb_epochs):
    # shuffle each epoch? 
    epoch_loss = 0
    count = 0
    for sample in dataloader:
        optimizer.zero_grad()  # correct place?
        # Will the order be important? No if fully connected network
        s = sample[:, [3,4,6,7]]  # x[0]^n, x[1]^n
        dsdt_approx = model(s)
        try:
            snew = s[:,[1,3]] + dt * dsdt_approx  # One step Euler
        except:
            #print("break: size s: ", s.shape)
            # incorrect batch size (not sure actually)
            quit()
            break
        s_exact = sample[:, [5,8]]
        s_approx = snew
        loss = loss_fn(snew, s_exact)
        epoch_loss += loss.item()
        loss.backward()
        losses.append(loss)
        optimizer.step()
        count += 1
    epoch_losses.append(epoch_loss / count)
print("epoch_losses: ", len(epoch_losses), epoch_losses)

# With a trained NN, choose some initial condition and draw a trajectory (x, y)

# The network is not training when I input sequences into the NN!

dt1 = dt #* 0.5

sv0 = torch.tensor([1., 0.])  # at t = 0
# The exact solution is x = cos(t), y = sin(t)
sv1 = torch.tensor([np.cos(dt), np.sin(dt1)])
# sol.y[0] is y[0], sol.y[1] is y[1]
# sv: first the point at t^n, followed by the point at t^{n+1}
sv = torch.tensor([*sv0, *sv1], dtype=torch.float32)   # y[0] and y[1] are interleaved (y[0][0], y[1][0], y[0][1], y[1][1])
x_lst = []
y_lst = []
t_lst = []

print("sv: ", sv)
# The model requires three points. This means I must find a way to compute the first two points. I will consider the exact solution. 
# In reality, I could apply two steps of a Runga-Kutta algorithms, or several steps of Euler with smaller time step. 

# Save initial condition (x,y,t) (same as y[0], y[1], t)
x_lst.append(sv0[0].item())
y_lst.append(sv0[1].item())
t_lst.append(0)
print("x_lst= ", x_lst)

sv_nm1 = sv0.clone()
sv_n   = sv1.clone()
sv = torch.tensor([*sv_nm1, *sv_n], dtype=torch.float32)   # y[0] and y[1] are interleaved (y[0][0], y[1][0], y[0][1], y[1][1])

# The method is unstable
for i in range(600):
    # My sv update is incorrect
    dydt = model(sv)  # model takes in first two points. How is that done? 
    #sv_new = sv[2:] + dt1 * dydt
    sv_np1 = sv_n + dt1 * dydt  
    sv_nm1 = sv_n.clone()  # Inefficient, but clear, implementation
    sv_n   = sv_np1.clone()
    sv = torch.tensor([*sv_nm1, *sv_n], dtype=torch.float32)
    x_lst.append(sv_np1[0].item())
    y_lst.append(sv_np1[1].item())
    t_lst.append((i+1)*dt1)

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

