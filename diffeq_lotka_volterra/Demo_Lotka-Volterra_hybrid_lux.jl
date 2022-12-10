using Lux, DiffEqFlux
using DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
# Potential others required: Optimizers.jl, Zygote.jl
using Optimisers

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

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

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

# u = [α*🐇, -δ*🐺]

function circleODEfunc(du, u, p, t)
    du[1] = -u[2]
    du[2] =  u[1]
    return du
end

p = [1.5, 1., 3., 1.]

# ODEfunc = trueODEfunc
# ODEfunc = circleODEfunc
ODEfunc = lotka_volterra!

#α, β, γ, δ = p  # accessing named tuple (WHY?)
β = 1.
γ = 3.

prob_trueode = ODEProblem(ODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
dense = Lux.Dense

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

# Custom layer is necessary to model the linear part of the equation
struct LuxLinear5 <: Lux.AbstractExplicitLayer
  β::Float64
  γ::Float64
end

# LuxLinear = LuxLinear5

function LuxLinear5(a::Float32, b::Float32)
  println("hello")
  return LuxLinear5(() -> copy(a), () -> copy(b))
end

lin_layer = LuxLinear5(β, γ)
Lux.initialstates(rng::AbstractRNG, layer::LuxLinear5) = (β=layer.β,γ=layer.γ)
Lux.initialparameters(rng::AbstractRNG, layer::LuxLinear4) = ()

# define layer
(l::LuxLinear5)(x, st) = (-st.β * x[1], st.γ * x[2])

mmm = LuxLinear5(0.3, 0.4)
function construct_model(n, dense)
  # α, β, γ, δ = p 
  # println("beta: ", β)
  β = 1.
  γ = 3.
  # y1 = x[1]
  # y2 = x[2]
  layer1 = Lux.Chain(dense(2, n, tanh), dense(n, 2))
  # layer2 = Lux.Chain(x -> x)  # x -> [-x[2], x[1]]
  # Once compiled, I cannot change definitions? HOW TO DEBUG? 
  layer2 = Lux.Chain(LuxLinear4((β,γ)]))  
  return Lux.PairwiseFusion(+; layer1, layer2)  # + does not work with sequence
end

dudt2 = construct_model(n, Lux.Dense)

# Output parameters and status
p, st = Lux.setup(rng, dudt2)
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
