
# Whether all modules are loaded correctly apparently depends on the order they are loaded. WHY? 
# If I shift-enter 10 lines at a time, I sometimes get some failures. Is there any way for the wrong
# order to be prevented via compiler analysis? 
#using OrdinaryDiffEq
using DifferentialEquations
using ModelingToolkit  # for @variables
using DataDrivenDiffEq
using DataDrivenSparse
using LinearAlgebra
using SciMLSensitivity #<<<
using Optimization #<<<
using OptimizationOptimisers #<<<
using OptimizationOptimJL #<<<
using ComponentArrays #<<<
#using DiffEqSensitivity
#using Optim
# using DiffEqFlux
using Lux
using Plots
gr()
using Statistics
# Set a random seed for reproduceable behaviour
using Random
Random.seed!(1234)

# Create a name for saving ( basically a prefix )
svname = "Scenario_1_"

## Data generation
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
    return du
end

#p_ = p_LV = Float32.([1.3, 0.9, 0.8, 1.8])
p_ = p_LV = [1.3, 0.9, 0.8, 1.8]
tspan = (0.0f0,3.0f0)  # must be a tuple (works as a list outside function, not inside)
u0 = Float32.([0.44249296,4.6280594])
u0 = [0.44249296,4.6280594]
saveat=0.1

function generate_solution_ode(system_odes, u0, tspan, p_; saveat=0.1)
    prob = ODEProblem(system_odes, u0,tspan, p_)
	# make sure error tolerances are not set too low. If they are too low, the 
	# integration might complete early. 
    solution = solve(prob, Vern7(), abstol=1e-7, reltol=1e-4, saveat=0.1)
    return solution
end

solution = generate_solution_ode(lotka!, u0, tspan, p_LV)
# Ideal data
X = Array(solution)
t = solution.t
plot(X')

# Add noise in terms of the mean (Statistics.jl). In this way, less noise is added
# if the mean is lower
x̄ = mean(X, dims = 2)
noise_magnitude = Float32(5e-2)
Xₙ = X .+ (noise_magnitude*x̄) .* randn(eltype(X), size(X))

# Turn plotting of noisy solution on/off.
if 1 == 0
    plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
    scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])
end

## ------------------ Define the UDE problem ---------------------------
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))  # Why? 

# Try the networks with different sizes, different depths, and different activation functions 
# Activation functions to try: rbf, tanh, relu, elu, sigmoid

# Define the network 2->5->5->5->2
# DiffEqFlux.FastChain (more efficient than Chain for smaller networks)
#=
inner_layers = []
for i in 1:nb_inner_layers
    push!(inner_layers, FastDense(layer_size, layer_size, rbf))
end

U = FastChain(
        FastDense(2,layer_size,rbf), 
        inner_layers...,
        FastDense(layer_size,2)
)
=#

#= =#

# Set a random seed for reproduceable behaviour
rng = Random.default_rng()
Random.seed!(1234)

function create_model(act, rng; layer_size=5)
    FastDense = Lux.Dense
    U = Lux.Chain(
        FastDense(2, layer_size, act), 
        FastDense(layer_size, layer_size, act), 
        FastDense(layer_size, layer_size, act), 
        FastDense(layer_size,2))
    p, st = Lux.setup(rng, U)
    return U, p, st
end

layer_size = 5
nb_inner_layers = 2
U, p_NN, st = create_model(rbf, rng; layer_size=layer_size)

# Define the hybrid model
function ude_dynamics!(du,u, p, t, p_LV, U)
    û = U(u, p) # Network prediction
    du[1] = p_LV[1]*u[1] + û[1]
    du[2] = -p_LV[4]*u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du,u,p,t) = ude_dynamics!(du,u,p,t,p_NN, U)
# Define the problem
# 2nd argument are initial conditions
prob_nn = ODEProblem(nn_dynamics!,Xₙ[:, 1], tspan, p_LV)

#=----
function predict(θ, X = Xₙ[:,1], T = t, scheme)
    Array(solve(prob_nn, scheme(), u0 = X, p=θ,
                tspan = (T[1], T[end]), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = DiffEqFlux.ForwardDiffSensitivity()
                ))
end
=#

## Function to train the network
# Define a predictor (NODE) -------------------------
# QUESTION: why no semi-colon? 
function predict(θ, X = Xₙ[:,1], T = t)
    _prob = remake(prob_nn, u0=X, tspan=(T[1],T[end]), p=θ)
    Array(solve(_prob, Vern7(), u0 = X, p=θ,
                tspan = (T[1], T[end]), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = DiffEqFlux.ForwardDiffSensitivity()
                ))
end

# Simple L2 loss
function loss(θ)
    X̂ = predict(θ)
    sum(abs2, Xₙ .- X̂)
end

# Container to track the losses
losses = Float64[]  # Code crashes without this line (in sciml_train: losses not defined)

# Callback to show the loss during training
callback = function(θ,l) 
    push!(losses, l)
    if length(losses)%50==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

## Training ------------------------------
:wait
# First train with ADAM for better convergence -> move the parameters into a
# favourable starting positing for BFGS
adtype = Optimization.AutoZygote()
# @which  OptimizationProblem is in SciMLBase (why not in Optimization?)
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p_NN))
println("before 1st Diffeqflux, length(losses): ", length(losses))
# res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.1f0), cb=callback, maxiters = 200)
res1 = Optimization.solve(optprob, ADAM(0.1), callback=callback, maxiters=200)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
# Train with BFGS
println("before 2nd Diffeqflux, length(losses): ", length(losses))
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 10000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")
### ERROR  on the first sciml_train() above ######### WHY? 
# ===================================================================================
## Plot the losses
println("before plotting, length(losses): ", length(losses))
pl_losses = plot(1:200, losses[1:200], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(201:length(losses), losses[201:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
# savefig(pl_losses, joinpath(pwd(), "plots", "$(svname)_losses.pdf"))
# Rename the best candidate
# where does minimizer come from? 
p_trained = res2.minimizer

## Analysis of the trained network
# Plot the data and the approximation
# Predict u[1], u[2]  (solution at all times)
t̂ = t[1]:0.05f0:t[end]
X̂ = predict(p_trained, Xₙ[:,1], t̂)
# Trained on noisy data vs real solution
pl_trajectory = plot(t[1]:0.05f0:t[end], transpose(X̂), xlabel = "t", ylabel ="x(t), y(t)", color = :red, label = ["UDE Approximation" nothing])
scatter!(t, transpose(Xₙ), color = :black, label = ["Measurements" nothing])
# savefig(pl_trajectory, joinpath(pwd(), "plots", "$(svname)_trajectory_reconstruction.pdf"))

## ----------------
# n_tspan = (0.0f0,4.8f0)
# t = n_tspan[1]:0.1f0:n_tspan[end]
# prob_nn_oz = ODEProblem(nn_dynamics!, u0, n_tspan, res2.minimizer)
# sol_nn_oz = predict(p_trained, Xₙ[:,1], t)

# lot_oz = ODEProblem(lotka!, u0, n_tspan, p_)
# sol_oz = solve(lot_oz, saveat=0.1)

# plot(sol_oz)
# scatter!(t, transpose(sol_nn_oz))
##

# Ideal unknown interactions of the predictor (modelled portion of PDE)
# The solution is defined at times given by t̂
Ȳ = [-p_[2]*(X̂[1,:].*X̂[2,:])' ; p_[3]*(X̂[1,:].*X̂[2,:])']
# Neural network guess (modelled terms in ODE)
Ŷ = U(X̂, p_trained)

# Error is fairly large, reaching almost 1 ( solution of 4-7)
pl_reconstruction = plot(t[1]:0.05f0:t[end], transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["UDE Approximation" nothing])
plot!(t[1]:0.05f0:t[end], transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
# savefig(pl_reconstruction, joinpath(pwd(), "plots", "$(svname)_missingterm_reconstruction.pdf"))

# Plot the error
# norm not found. 
pl_reconstruction_error = plot(t[1]:0.05f0:t[end], norm.(eachcol(Ȳ-Ŷ)), yaxis = :log, xlabel = "t", ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2,1))
# savefig(pl_missing, joinpath(pwd(), "plots", "$(svname)_missingterm_reconstruction_and_error.pdf"))
pl_overall = plot(pl_trajectory, pl_missing)
# savefig(pl_overall, joinpath(pwd(), "plots", "$(svname)_reconstruction.pdf"))

#------------------------------------------------------------
function setupFunction(; n=100)
    g(u,v) = -1 .+ (2.5 .* u.^3) .+ (u .* v) .+ (v .^ 2)
    # h(u,v) = g(u,v)  # The two functions can be the same
    h(u,v) = (2.5 .* v.^3) .+ (u .* v) .+ (u .^ 2)
    X = randn(n)
    Y = randn(n)
    Z1 = reshape(g(X,Y), 1, n)
    Z2 = reshape(h(X,Y), 1, n)
    Z = vcat(Z1, Z2)
    # X,Y must have at least two columns
    X = reshape(X, 1, n)
    Y = reshape(Y, 1, n)
    X = vcat(X, Y)
    println("return from setupFunction, size(X): ", size(X))
    println("-------------------------------------------")
    return X, Z
end

function sparseBasis(X, Y; n=100)
    problem = DirectDataDrivenProblem(X, Y, name=:test1)
    @variables u[1:2]
    u = collect(u)
    basis = Basis(polynomial_basis(u, 3), u)
    println("typeof(basis): ", typeof(basis))
    # X = Float64.(X)
    # Y = Float64.(Y)
    # println(basis)
    println("size(X,Y): ", size(X), size(Y))
    println("type(X, Y): ", typeof(X), typeof(Y))
    println("methods: ", @which solve(problem, basis, STLSQ()))
    res = solve(problem, basis, STLSQ())

    # What is the fit?
    println(res |> get_basis)
    println(res |> get_basis |> get_parameter_values)
    println("residual: ", res.residuals)
end

X = X̂
Y = Ŷ
# Without the conversion to Float64, I get an error related to set_active! not having proper signature
X = Float64.(X)
Y = Float64.(Y)
# println("typeof(X,Y): ", typeof(X), typeof(Y))
# X, Y = setupFunction(;n=61)
#X = Float32.(X)
#Y = Float32.(Y)
# println("typeof(X,Y): ", typeof(X), typeof(Y))
sparseBasis(X, Y; n=61)
#--------------------------------------------------------------
# Retune for better parameters -> we could also use DiffEqFlux or other parameter estimation tools here.
# Compute recovered analytical functions using basis. HOW TO DO THIS?

# Define the recovered, hyrid model
function recovered_dynamics!(du,u, p, t, p_true)
    û = Ψf(u, p) # Network prediction
    du[1] = p_true[1]*u[1] + û[1]
    du[2] = -p_true[4]*u[2] + û[2]
end

# Closure with the known parameter
estimated_dynamics!(du,u,p,t) = recovered_dynamics!(du,u,p,t,p_)

estimation_prob = ODEProblem(estimated_dynamics!, u0, tspan, p̂)
estimate = solve(estimation_prob, Tsit5(), saveat = t)

# Plot
plot(solution)
plot!(estimate)

## Simulation

# Look at long term prediction
t_long = (0.0f0, 20.0f0)
estimation_prob = ODEProblem(estimated_dynamics!, u0, t_long, p̂)
estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.1) # Using higher tolerances here results in exit of julia
plot(estimate_long)

true_prob = ODEProblem(lotka!, u0, t_long, p_)
true_solution_long = solve(true_prob, Tsit5(), saveat = estimate_long.t)
plot!(true_solution_long)

## Save the results
save(joinpath(pwd(), "results" ,"$(svname)recovery_$(noise_magnitude).jld2"),
    "solution", solution, "X", Xₙ, "t" , t, "neural_network" , U, "initial_parameters", p, "trained_parameters" , p_trained, # Training
    "losses", losses, "result", Ψf, "recovered_parameters", p̂, # Recovery
    "long_solution", true_solution_long, "long_estimate", estimate_long) # Estimation


## Post Processing and Plots

c1 = 3 # RGBA(174/255,192/255,201/255,1) # Maroon
c2 = :orange # RGBA(132/255,159/255,173/255,1) # Red
c3 = :blue # RGBA(255/255,90/255,0,1) # Orange
c4 = :purple # RGBA(153/255,50/255,204/255,1) # Purple

p1 = plot(t,abs.(Array(solution) .- estimate)' .+ eps(Float32),
          lw = 3, yaxis = :log, title = "Timeseries of UODE Error",
          color = [3 :orange], xlabel = "t",
          label = ["x(t)" "y(t)"],
          titlefont = "Helvetica", legendfont = "Helvetica",
          legend = :topright)

# Plot L₂
p2 = plot3d(X̂[1,:], X̂[2,:], Ŷ[2,:], lw = 3,
     title = "Neural Network Fit of U2(t)", color = c1,
     label = "Neural Network", xaxis = "x", yaxis="y",
     titlefont = "Helvetica", legendfont = "Helvetica",
     legend = :bottomright)
plot!(X̂[1,:], X̂[2,:], Ȳ[2,:], lw = 3, label = "True Missing Term", color=c2)

p3 = scatter(solution, color = [c1 c2], label = ["x data" "y data"],
             title = "Extrapolated Fit From Short Training Data",
             titlefont = "Helvetica", legendfont = "Helvetica",
             markersize = 5)

plot!(p3,true_solution_long, color = [c1 c2], linestyle = :dot, lw=5, label = ["True x(t)" "True y(t)"])
plot!(p3,estimate_long, color = [c3 c4], lw=1, label = ["Estimated x(t)" "Estimated y(t)"])
plot!(p3,[2.99,3.01],[0.0,10.0],lw=1,color=:black, label = nothing)
annotate!([(1.5,13,text("Training \nData", 10, :center, :top, :black, "Helvetica"))])
l = @layout [grid(1,2)
             grid(1,1)]
plot(p1,p2,p3,layout = l)

savefig(joinpath(pwd(),"plots","$(svname)full_plot.pdf"))
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
