using DifferentialEquations
using Plots
using Flux
using DiffEqFlux

function lotka_volterra(du, u, p, t)
    x, y = u
    α, β, γ, δ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

u0 = [1., 1.]
tspan = [0., 10.]
p = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lotka_volterra, u0, tspan, p)

# Solve equation 
sol = solve(prob)
plot(sol)
#------------------------------------------
# Make initial condition (u0) and time spans (tspan) a function 
# of the parameters
u0_f(p, t0) = [p[2], p[4]]  # coefficients of x*y
tspan_f(p) = (0., 10*p[4])
p = [1.5, 1., 3., 1.]
prob = ODEProblem(lotka_volterra, u0_f, tspan_f, p)
sol = solve(prob)
plot(sol)
#--------------------------------------------
# Put the ODE into a NN

p = [1.5, 1.1, 3.0, 1.0]
prob = ODEProblem(lotka_volterra, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=0.1)
plot(sol)
A = sol[1,:]
plot(sol)
t = 0:.1:100   # Use the times the solution is saved at. No automatic interpolation onto new times
scatter!(t, A)    
#---------------------------------------------------

p = [2.2, 1.0, 2.0, 0.4]
# Add p to parameters to optimize
params = Flux.params(p)

function predict_rd()
    solve(prob, Tsit5(), p=p, saveat=0.1)[1,:]  #u[1] as a function of t
end

# Initial loss
loss_rd() = sum(abs2, x-1 for x in predict_rd())

# predict_rd()
# plot(predict_rd())
# loss_rd()

# Train Flux to minimize the loss (the neural network is a parameterized function)
# data: iteration of size 100, elements are all ()
data = Iterators.repeated((), 100)

#Explicit Euler runs with dt=0.01 up to t=10. 
opt = ADAM(0.1)
cb = function()   # callback
    display(loss_rd())
    #display(plot(solve(remake(prob, p=p), Euler(), dt=0.01, saveat=0.1), ylim=(0,6)))
    display(plot(solve(remake(prob, p=p), Tsit5(), saveat=0.1), ylim=(0,6)))
end

cb()
# Note: p has not changed. Why? 
#------------------------------------------------

# Train with Flux
Flux.train!(loss_rd, params, data, opt, cb = cb)

params
loss_rd()

m = Chain(
    Dense(28^2, 32, relu),
    Dense(32, 10),
    softmax
)

# The Flux training above does not seem to work as described in the demo at
# https://julialang.org/blog/2019/01/fluxdiffeq/
#---------------------------------------------------------
m