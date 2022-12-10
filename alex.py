import torch 
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from IPython import embed # to exit, type "%exit_raise"

class Net(nn.Module):
    # 2 layers
    def __init__(self, dim1, dim2, dim3):

        super(Net, self).__init__()
        self.layer_in = nn.Linear(2, dim1)
        self.layer_hid1 = nn.Linear(dim1, dim2)
        self.layer_hid2 = nn.Linear(dim2, dim3)
        self.layer_out = nn.Linear(dim3, 2)
        #self.active = nn.ReLU()
        #self.active = nn.Sigmoid()
        self.active = nn.Tanh()

    def forward(self, xy):
        embed()
        xy = self.active(self.layer_in(xy))
        xy = self.active(self.layer_hid1(xy))
        xy = self.active(self.layer_hid2(xy))
        xy = self.layer_out(xy)
        
        return xy
    
    def func(self, val):
        pass
        #val = torch.tensor([float(val)])
        #ans = self(val)
        #return float(ans)

def sys_eq(t,x):

    alpha = 1.
    beta  = 1.
    gamma = 1.
    delta = 1.

    dxdt    = np.zeros([2,1])
    dxdt[0] = alpha*x[0] - beta*x[0]*x[1]  # dx/dt
    dxdt[1] = delta*x[0]*x[1] - gamma*x[1] # dy/dt

    return dxdt



def main():
    alpha = 1.
    beta  = 1.
    gamma = 1.
    delta = 1.


    npts = 100

    # t ∈ [t0, tn]
    tn = 40
    t0 = 0
    t = np.linspace(t0, tn, npts)

    # Initial Values
    x0 = [10., 5.]

    # Solve ODE (Credit: [1])
    xy = np.zeros((len(t), len(x0)))
    xy[0,:] = x0
    r = integrate.ode(sys_eq).set_integrator("dopri5")
    r.set_initial_value(x0, t0)
    for i in range(1, t.size):
        xy[i, :] = r.integrate(t[i]) # get one more value, add it to the array
        if not r.successful():
            raise RuntimeError("Could not integrate")

    x = xy[:,0]
    y = xy[:,1]

    # Training data (NN: x,y -> F,G)
    F = -beta * x * y
    G = delta * x * y

    # Normalize
    maxx = max(x)
    maxy = max(y)
    maxF = max(F)
    maxG = max(G)
    x = x / maxx
    y = y / maxy
    F = F / maxF
    G = G / maxG

    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    F = torch.tensor(F, dtype=torch.float)
    G = torch.tensor(G, dtype=torch.float)
    x = x.reshape(len(x),1) # Column vector
    y = y.reshape(len(y),1) # Column vector
    F = F.reshape(len(F),1) # Column vector
    G = G.reshape(len(G),1) # Column vector

    # Create a all-to-all feed-forward network with 3 layers
    dim1 = 300 # Amount of nodes in hidden layer 1
    dim2 = dim1 # Amount of nodes in hidden layer 2
    dim3 = dim1 # Amount of nodes in hidden layer 3
    model = Net(dim1, dim2, dim3)
    
    #quit()
    #------------------------------------------------------------------------------------------------------

    loss_fct = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.e-3, weight_decay=1.e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1.e-3)
    n = 1000
    t = np.linspace(t0,tn*2,n)
    F_nn = np.zeros(n)
    G_nn = np.zeros(n)

    max_epochs = 8000
    for epoch in range(1, max_epochs):

        Fpred, Gpred = model(x,y)
        
        # calculate the loss
        loss = loss_fct(Fpred, Gpred, y.reshape(len(y),1))

        if epoch % 10 == 0:
            print("loss: ", loss.detach().item())
            
        optimizer.zero_grad()
        loss.backward()  # Backpropagataion
        optimizer.step()

    n = 1000
    t = np.linspace(x0*2,xn*2,n)
    y_nn = np.zeros(n)
    for i in range(n):
        y_nn[i] = model.func(t[i]) * maxy # Un-normalize
    y_real = f(t)
    
    plt.rcParams['text.usetex'] = True # Use LaTeX
    plt.rcParams.update({'font.size': 16})
    
    print("before plt")

    plt.figure()
    #plt.scatter(x, y*maxy, label='Noisy data')
    plt.plot(t, y_nn, 'r-', label='Neural Net')
    plt.plot(t, y_real, 'b--', label='Real plot')
    plt.legend()
    
    plt.show()
    
#----------------------------------------------------------------------
if __name__ == '__main__':
    main()
