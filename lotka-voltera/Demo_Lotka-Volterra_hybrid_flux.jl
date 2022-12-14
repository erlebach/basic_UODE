using DiffEqFlux
using DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
# Potential others required: Optimizers.jl, Zygote.jl
using Optimisers

# Implementation using custom layer in Flux (I could not get Lux to work). 

# Use NODE to solve circle equations:
#  dx/dt = -y 
#  dy/dt =  NN
#  where NN is x
#  Once I can do that, I will construct more complex systems

rng = Random.default_rng()
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 7.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

# function trueODEfunc(du, u, p, t)
#     true_A = [-0.1 2.0; -2.0 -0.1]
#     du .= ((u.^3)'true_A)'
# end

"""
du/dt = A u + f(u)
Input to NN: u (2-d vector)
Output: is du/optimize_datetime_ticks

I want: input : u
Output: f(u)
"""

function lotka_volterra!(du, u, p, t)
  🐇, 🐺 = u
  α, β, γ, δ = p 
  du[1] = d🐇 = α*🐇 - β*🐇*🐺
  du[2] = d🐺 = γ*🐇*🐺 - δ*🐺
  return du
end

function lotka_volterra_NN!(du, u, p, t, NN)
  #=
     NN: output of a in-2 ==> out-2 neural network
     return: estimate of time derivative at time tⁿ
  =#
  🐇, 🐺 = u    # solution at time tⁿ
  α, β, γ, δ = p 
  du[1] = d🐇 = α*🐇 + NN₁
  du[2] = d🐺 = NN₂ - δ*🐺
  return du
end

p = [1.5, 1., 3., 1.]

ODEfunc = lotka_volterra!
NODEfunc(du, u, p, t) = lotka_volterra_NN!(du, u, p, t, model)

#α, β, γ, δ = p  # accessing named tuple (WHY?)
β = 1.
γ = 3.

prob_trueode = ODEProblem(ODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dense = Lux.Dense
chain = Lux.Chain

##### 
# Use Lux.BranchLayer(x)  (pass x to each layer
# First layer will be linear layer (no parameters)
# Second layer will be a NN Chain that will model the nonlinear term in the equation)
# The sum of the two will be du/df for the full equation)
# Both layers will return a vector of size 2
# Need a function to add two layers together
# 
#dudt2 = Lux.Chain(x -> x.^3,
n = 30

# Lux model to handle the nonlinear term in the Lotka-Volterra equations
layer_size = 5
act = Lux.tanh

function construct_model()
  model = chain(dense(2, layer_size, act),
              dense(layer_size, layer_size, act),
              dense(layer_size, layer_size, act),
              dense(layer_size, 2))
    return model
end

model = construct_model()
rng = Random.default_rng()
Random.seed!(rng, 0)
# ps are parameters
# st are state (frozen)
ps, st = Lux.setup(rng, model)

dudt2 = model

# Output parameters and status
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
  Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end


# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback13 = function (p, l, pred; doplot = false)
  println(l)
  println("before if statement, Display callback plot")
  # plot current prediction against data
  if doplot
    plt = scatter(tsteps, ode_data[1,:], label = "data")
    scatter!(plt, tsteps, pred[1,:], label = "prediction")
    display(plot(plt))
    println("Display callback plot")
  end
  return false
end

pinit = Lux.ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob,
                                       ADAM(0.05),
                                       callback = callback13,
                                       maxiters = 10)

optprob2 = remake(optprob,u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
                                        Optim.BFGS(initial_stepnorm=0.01),
                                        callback=callback,
                                        allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)
