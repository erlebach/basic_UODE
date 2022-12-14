# Follow demo at https://docs.sciml.ai/Overview/dev/showcase/missing_physics/

using Revise
using Lux, DiffEqFlux
using DifferentialEquations, Optimization, OptimizationOptimisers, OptimizationOptimJL, Random, Plots
# Potential others required: Optimizers.jl, Zygote.jl
using Optimisers
using Statistics, Random, LinearAlgebra
using ComponentArrays #for lux
using ModelingToolkit, DataDrivenDiffEq, DataDrivenSparse

# Use NODE to solve circle equations:
#  dx/dt = -y 
#  dy/dt =  NN
#  where NN is x
#  Once I can do that, I will construct more complex systems

#p = [1.5, 1., 3., 1.]
#u0 = Float32[2.0; 0.0]
# Follow demo at https://docs.sciml.ai/Overview/dev/showcase/missing_physics/
p_LV = [1.3, 0.9, 0.8, 1.8]  # Chris-Rackauckas demo
rng = Random.default_rng()
u0_LV = [0.44249296,4.6280594] # Chris-Rackauckas demo
datasize = 30
tspan = (0.0f0, 3.0f0)
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

ODEfunc = lotka_volterra!

# calling ODEProblem with p=p as last argument will not work in Julia. 
prob_trueode = ODEProblem(ODEfunc, u0_LV, tspan, p_LV)
# Make sure error tolerances are not too low, or else maxiters might have to be increased (from C.R.)
solution = solve(prob_trueode, Vern7(), abstol=1.e-7, reltol=1.e-7, saveat=0.1)

# Ideal (non-noisy) data
X = Array(solution)
t = solution.t
DX =Array(solution(solution.t, Val{1}))

# Add noise in terms of the mean
x̄ = mean(X, dims=2)  # mean along the 2nd dimension
noise_magnitude = 2.e-2
Xₙ = X .+ (noise_magnitude*x̄) .* randn(eltype(X), size(X))

plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])

#---------------------------------------------------------------------------
# Define the Neural Netowrk model
#α, β, γ, δ = p  # accessing named tuple (WHY?)


function construct_model(layer_size, act)
    dense = Lux.Dense
    chain = Lux.Chain

    model = chain(dense(2, layer_size, act),
              dense(layer_size, layer_size, act),
              dense(layer_size, layer_size, act),
              dense(layer_size, 2))
    return model
end

rbf(x) = exp.(-(x.^2))
act = tanh
layer_size = 10 

dudt2 = model = construct_model(layer_size, rbf)
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, model)


function lotka_volterra_NN!(du, u, p_NN, t, p_LV) 
  #=
     NN: output of a in-2 ==> out-2 neural network
     return: estimate of time derivative at time tⁿ
  =#
  🐇, 🐺 = u    # solution at time tⁿ
  α, β, γ, δ = p_LV 
  û = model(u, p_NN, st)[1]  # Network prediction (WHY [1]?)
  du[1] = d🐇 = α*🐇 + û[1]
  du[2] = d🐺 = û[2] - δ*🐺
  return du
end

#Define the hybrid model
# Missing

neuralODEfunc(du, u, p, t) = lotka_volterra_NN!(du, u, p, t, p_LV)

#define the problem with noisy solution as initial conditions
prob_nn = ODEProblem(neuralODEfunc, Xₙ[:, 1], tspan, p)

## Function to train the NN 
# define a predictor
function predict_neuralode(θ, X=Xₙ[:,1], T=t)
  # println("before remake")
  # println("T: ", T) # All the intermediate T
  # Chris does not the ";"
  _prob = remake(prob_nn; u0=X, tspan= (T[1], T[end]), p=θ)
  # println("after remake")
  Array(solve(_prob, Vern7(), saveat= T, 
      abstol=1.e-6, reltol=1.e-6, sensealg=ForwardDiffSensitivity()
  # Returns last computed element
  ))
end

# Simple L2 loss
function loss_neuralode(θ)
    # println("params: ": θ)
    X̂ = predict_neuralode(θ)
    sum(abs2, Xₙ .- X̂)
end

# Track the losses
losses = Float64[]


# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
# INEFFICIENCY: losses is an array in the global space. Why does the 
# callback work correctly when called from inside some other function. 
# Does some other function "see" `losses`? 
callback = function(p, loss) #, pred; doplot = false)
  # Printing to stdout leads to an erro when executing 
  # println("enter callback, p: $(p)")
  push!(losses, loss)
  if length(losses) % 50 == 0
    println("Current loss after $(length(losses)) iterations: $(losses[end])")
  end
  return false

  #=
  # plot current prediction against data
  if doplot
    plt = scatter(tsteps, ode_data[1,:], label = "data")
    scatter!(plt, tsteps, pred[1,:], label = "prediction")
    display(plot(plt))
    println("Display callback plot")
  end
  return false
  =#
end

#-------------------------------------------------------------------
# Train the network

# First train with ADAM for better convergence. This places the parameters 
# in a favourable starting position for BFGS. 

# Question: What integrator to use for ODE? Same as ideal problem? NO, it is defined in predict_neuralode

adtype = Optimization.AutoZygote()
# adtype = Optimization.AutoFiniteDiff()
# ERROR: Cannot find function signature. Must wait on Rackauckas
optf = Optimization.OptimizationFunction((x,p)->loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
# The losses are not changing <<<
# Adding print statements to certain functions creates problems with Zygote
# I do not get printout every 50 steps
res1 = Optimization.solve(optprob, ADAM(0.1), callback=callback, maxiters=200)
println("Training loss initially: $(losses[1])")
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.minimizer)
# I get printout every 50 steps
# Why does Optimization oslve slow down the longer it runs? Ths would indicate that certain computations are getting more expensive
# What are the stopping criteria?
# How does one use batches?
# How to interrupt running code? ctrl-C is not working from REPL
res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01), callback=callback, maxiters = 500)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Rename the best candidate
p_trained = res2.minimizer
#-------------------------------------------------------------
# Plot the losses
pl_losses = plot(1:200, losses[1:200], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(201:length(losses), losses[201:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
#------------------------------------------------------------------
## Analysis of the trained network
# Plot the data and the approximation
ts = first(solution.t):mean(diff(solution.t))/2:last(solution.t)
X̂ = predict_neuralode(p_trained, Xₙ[:,1], ts)
# Trained on noisy data vs real solution
pl_trajectory = plot(ts, transpose(X̂), xlabel = "t", ylabel ="x(t), y(t)", color = :red, label = ["UDE Approximation" nothing])
scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Measurements" nothing])
#----------------------------------------------------------------------
# Ideal unknown interactions of the predictor
Ȳ = [-p_LV[2]*(X̂[1,:].*X̂[2,:])';p_LV[3]*(X̂[1,:].*X̂[2,:])']
# Neural network guess
Ŷ = model(X̂,p_trained,st)[1]
#----------------------------------------------------------------------
# The reconstructin is really poor
pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["UDE Approximation" nothing])
# ideal
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
#-----------------------------------------------------------------------
# Plot the error
pl_reconstruction_error = plot(ts, norm.(eachcol(Ȳ-Ŷ)), yaxis = :log, xlabel = "t", ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2,1))
#-----------------------------------------------------------------------
# Symbolic regression via sparse regression (SINDy based)
# Create a Basis
@variables u[1:2]
# Generate the basis functions, multivariate polynomials up to deg 5
# and sine
basis = Basis(polynomial_basis(u,5), u);
push!(basis, sin(u[1]))
push!(basis, sin(u[2]))
#------------------------------------------------------------------------
# Create the thresholds which should be used in the search process
λ = exp10.(-3:0.01:5)
# Create an optimizer for the SINDy problem
opt = STLSQ(λ)
# Define different problems for the recovery
ideal_problem = DirectDataDrivenProblem(X̂, Ȳ)
nn_problem = DirectDataDrivenProblem(X̂, Ŷ)
# Test on ideal derivative data for unknown function ( not available )

println("Sparse regression")
full_problem = DataDrivenProblem(X, t = t, DX = DX)
# Next line has an error: "u not defined". WHY?
# How to stop early?
full_res = solve(full_problem, basis, opt, maxiter = 10000, progress = true)
#------------------------------------------------------------------------
# What is the difference between full_problem, ideal_problem, and nn_problem? 
# non-noisy, noisy, nn search?
ideal_res = solve(ideal_problem, basis, opt, maxiter = 10000, progress = true)
# DataSampler and Batcher not found
#nn_res = solve(nn_problem, basis, opt, maxiter = 10000, progress = true, sampler = DataSampler(Batcher(n = 4, shuffle = true)))
nn_res = solve(nn_problem, basis, opt, maxiter = 10000, progress = true)
# Store the results
results = [full_res; ideal_res; nn_res]
#------------------------------------------------------------------------
# Show the results
map(println, results)
# Show the results
map(println ∘ result, results)
# Show the identified parameters
map(println ∘ parameter_map, results)
#------------------------------------------------------------------------
# Define the recovered, hybrid model
function recovered_dynamics!(du,u, p, t)
  û = nn_res(u, p) # Network prediction
  du[1] = p_[1]*u[1] + û[1]
  du[2] = -p_[4]*u[2] + û[2]
end

estimation_prob = ODEProblem(recovered_dynamics!, u0_LV, tspan, parameters(nn_res))
estimate = solve(estimation_prob, Vern7(), saveat = solution.t)

# Plot
plot(solution)
plot!(estimate)
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
