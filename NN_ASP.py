import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import embed  # to exit: "%exit_raise"
from scipy import integrate
from torch import nn


class Net(nn.Module):
    # 2 layers
    def __init__(self, hid_neurons):

        super(Net, self).__init__()
        self.layer_in = nn.Linear(2, hid_neurons)
        self.layer_hid1 = nn.Linear(hid_neurons, hid_neurons)
        self.layer_hid2 = nn.Linear(hid_neurons, hid_neurons)
        self.layer_hid3 = nn.Linear(hid_neurons, hid_neurons)
        # self.layer_hid4 = nn.Linear(hid_neurons, hid_neurons)
        # self.layer_hid5 = nn.Linear(hid_neurons, hid_neurons)
        # self.layer_hid6 = nn.Linear(hid_neurons, hid_neurons)
        self.layer_out = nn.Linear(hid_neurons, 2)
        #self.active = nn.ReLU()
        #self.active = nn.Sigmoid()
        self.active = nn.Tanh()

    def forward(self, xy):
        out = self.active(self.layer_in(xy))
        out = self.active(self.layer_hid1(out))
        out = self.active(self.layer_hid2(out))
        out = self.active(self.layer_hid3(out))
        # out = self.active(self.layer_hid4(out))
        # out = self.active(self.layer_hid5(out))
        # out = self.active(self.layer_hid6(out))
        out = self.layer_out(out)
        
        return out
    
    def func(self, val):
        pass
        #val = torch.tensor([float(val)])
        #ans = self(val)
        #return float(ans)

def custom_loss_function(output, target):
    '''
    output = F_NN and G_NN
    target = F    and G
    Equations:
        F_approx = dx/dt - αx
        G_approx = dy/dt + γxy
    '''
    print(output.shape, target.shape)
    
    loss = torch.mean((output - target)**2)
    return loss

#-------------------------------------------
def sys_eq(t,x):

    '''
    System of equations:
    dx/dt = αx  - βxy
    dy/dt = δxy - γy
    '''

    # These four lines are ok, but they are repeated each time this function is 
    # called, which is at every time step
    alpha = 1.5
    gamma = 1.
    beta  = 1.
    delta = 3.

    dxdt    = np.zeros([2,1])
    dxdt[0] = alpha*x[0] - beta*x[0]*x[1]  # dx/dt
    dxdt[1] = delta*x[0]*x[1] - gamma*x[1] # dy/dt

    return dxdt
#-------------------------------------------

def main():

    alpha = 1.
    beta  = 1.
    gamma = 1.
    delta = 1.

    npts = 200

    # t ∈ [t0, tn]
    t0, tn = [0., 10.]  # GE: I improved the notation
    t = np.linspace(t0, tn, npts)

    # Initial Values
    x0 = [1., 1.]   # GE: changed initial values (see link I sent you). Insert the link into your code

    # Solve ODE (Credit: [1])  # GE: put credits and comments at the top of the file to make sure people see them)
    xy = np.zeros((len(t), len(x0)))  # nb_t, nb_variables
    xy[0,:] = x0

    # "dopri5" is a RK45 method (Runge-Kutta)
    r = integrate.ode(sys_eq).set_integrator("dopri5")
    r.set_initial_value(x0, t0)


    for i in range(1, t.size):
        xy[i, :] = r.integrate(t[i]) # get one more value, add it to the array
        if not r.successful():
            raise RuntimeError("Could not integrate")

    x = xy[:,0]
    y = xy[:,1]

    print("shape xy: ", xy.shape)
    plt.plot(t, x)
    plt.plot(t, y)
    plt.show()
    plt.close()


    #-----------------------------------------------------
    # GE: You should choose random locations between 0 and 100 to find training data. 
    # Assume you take 50 points (out of 200) for training). You will need the point itself, 
    # and a point dt (dt is the time step) away, which will serve as a label against which 
    # the Euler method will be compared. (I am not sure why you changed my code to the extent
    # you did. 

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

    # x = torch.tensor(x, dtype=torch.float)
    # y = torch.tensor(y, dtype=torch.float)
    # x = x.reshape(len(x),1) # Column vector
    # y = y.reshape(len(y),1) # Column vector

    # Combine to make target vector
    FG = torch.zeros(F.shape[0], 2)
    xy = torch.zeros(x.shape[0], 2)
    FG[:,0] = torch.tensor(F)
    FG[:,1] = torch.tensor(G)
    xy[:,0] = torch.tensor(x)
    xy[:,1] = torch.tensor(y)
    
    # Create a all-to-all feed-forward network with 6 layers
    hid_dim = 100 # Amount of nodes in hidden layers
    model = Net(hid_dim)
    
    # loss_fct = nn.MSELoss(reduction='mean')
    loss_fct = nn.MSELoss(reduction='mean')
    # loss_fct = custom_loss_function

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.)#, weight_decay=1.e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1.e-3)

    # n = 1000
    # t = np.linspace(t0,tn*2,n)
    # t = torch.tensor(t, dtype=torch.float)
    # FG_NN = np.zeros([n,2])
    
    max_epochs = 8000
    for epoch in range(1, max_epochs):
        outpred = model(xy)
        
        # calculate the loss
        loss = loss_fct(outpred, FG) # shapes ok

        if epoch % 10 == 0:
            # plt.figure()
            # plt.plot(t,outpred)
            # plt.savefig('./img/fig'+str(epoch)+'.png')
            # plt.close()
            print("loss: ", loss.detach().item())
            
        optimizer.zero_grad()
        loss.backward()  # Backpropagataion
        optimizer.step()

    # For pyplot
    # plt.rcParams['text.usetex'] = True # Use LaTeX
    # plt.rcParams.update({'font.size': 16})

    '''
    TODO: Create plot here
    '''

#------------------------------------
if __name__ == "__main__":
    main()

'''
Credits:
[1]: https://stackoverflow.com/questions/48428140/imitate-ode45-function-from-matlab-in-python
'''
