# 2022-12-25
# Get the poly layer working on a simple example, and then perhaps integeate this into 
# standalone_lux_layer.jl
#
# 2022-12-25
# Solve the Lotka-Volterra equations where the nonlinear term is replaced by a polynomial layer. 
# The UODE will be trained, which will provide the polynomial structure on the right-hand side.

# Draw inspiration from  https://docs.sciml.ai/Overview/dev/showcase/missing_physics/ (2022-12-31)

using Revise
using Zygote
using ForwardDiff
using Lux
using Random
using Statistics
using Tullio
using Plots
# using Optimisers
using ComponentArrays
using LinearAlgebra
using Optimization
using OptimizationOptimJL
using DifferentialEquations
using SciMLSensitivity

# required to use Optimisers.jl
# See https://docs.sciml.ai/Optimization/stable/optimization_packages/optimisers/
using OptimizationOptimisers  

include("./Polynomial.jl")
include("./Polynomial_layer.jl")

# ================== END DEFINITION of Polynomial Layer =======================================

# Let us define a quadratic functions and apply the polynomial layer 
# to train the parameters

rng = Random.default_rng()
Random.seed!(rng, 0)

# Define the LV equations 
function Lotka_Volterra!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[1] * u[2]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

tspan = (0.f0, 6.0f0)
u0 = [0.44249296,4.6280594]
params = [1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(Lotka_Volterra!, u0, tspan, params)
solution = solve(prob, Vern7(), abstol=1.e-6, reltol=1.e-6; saveat=0.05)
t = solution.t  # solution steps

# Add noise in terms of the mean
X = Array(solution)
x̄ = mean(X, dims=2)  # mean along the 2nd dimension
noise_magnitude = 5.e-3
Xn = X .+ (noise_magnitude*x̄) .* randn(eltype(X), size(X))
plot(t, Xn[1,:])
plot!(t, Xn[2,:])

# Set up neural model for 2D polynomial layer
model = Polylayer(; out_dims=2, degree=2, init_weight=Lux.zeros32)
ps_NN, st = Lux.setup(rng, model)
ps_NN = ComponentArray(ps_NN)
opt = Optimisers.Adam(0.01f0)
st_opt = Optimisers.setup(opt, ps_NN)
Nt = length(solution.t)    # Not sure why N is required. 
x_data = Xn[1, :]
y_data = Xn[2, :]
z_data = missing  # z_data should be the experimental data. We won't use for now
dct = Dict(:rng => rng, :model => model, :ps_NN => ps_NN, :st => st, 
           :opt => opt, :st_opt => st_opt, :Nt => Nt, 
           :x_data => x_data, :y_data => y_data, :z_data => z_data)

# Define the LV equations with polynomial fitting
function Lotka_Volterra_NN!(du, u, p_NN, t, p_LV, model, st)
    # the model must act on all the points of the u solution. 
    # Currently, it is only acting on two elements. No way to fit the polynomial. 
    α, β, γ, δ = p_LV
    # model returns two polynomials
    û = model(u, p_NN, st)[1] # model also return state st. Keep 1st element. 
    du[1] =  α * u[1] + û[1]   # -α*u[1] + poly(acting on single point). Where is the network trained? 
    du[2] = -δ * u[2] + û[2]
    return du
end

# The 3rd argument: parameters of NN
p_LV = params
Lotka_Volterra_NN_closure(du, u, p_NN, t) = Lotka_Volterra_NN!(du, u, p_NN, t, p_LV, model, st)

prob_nn = ODEProblem(Lotka_Volterra_NN_closure, Xn[:, 1], tspan, ps_NN)

# Simple L2 loss
function loss_neuralode(θ, hyperparam, ps)
    X̂ = predict_neuralode(θ)  # should return solution
    λ = 0.001  # Induduce sparsity with L1 norm
    loss = sum(abs2, Xn .- X̂) + λ * norm(ps, 1) #/ length(ps)
    println("loss: ", loss)
    return loss
end
#-------------------------------------------------------------------------
#define the problem with noisy solution as initial conditions
# I do not understand which parameters should be included. Here we only include the NN parameters.
# Why aren't the parameters of the equation included? (could contatenate both parameter vectors)
prob_nn = ODEProblem(Lotka_Volterra_NN_closure, Xn[:, 1], tspan, ps_NN)

# T are the different times
# \theta are parameters of the network
function predict_neuralode(θ, X₀=Xn[:,1], T=t)
    # θ: parameters of NN

    println("θ[1] = ", θ.coeffs[:,1])
    println("θ[2] = ", θ.coeffs[:,2])

    # Update parameters
    _prob = remake(prob_nn; u0=X₀, tspan= (T[1], T[end]), p=θ)
    solution = solve(_prob, Vern7(), saveat= T,
        abstol=1.e-6, reltol=1.e-6, sensealg=ForwardDiffSensitivity()
        # is `sensealg` necessary?
    )

    solution = Array(solution)
    return solution 
end

losses = Float64[]

callback = function(p, loss) #, pred; doplot = false)
    push!(losses, loss)
    if length(losses) % 50 == 0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote() # best if many parameters
optf = Optimization.OptimizationFunction((x, p)->loss_neuralode(x, p, ps_NN), adtype)
# Componentarray is a projection operator
optprob = Optimization.OptimizationProblem(optf, ps_NN)
# The parameters are not changing. 
# res1 = Optimization.solve(optprob, Descent(0.001); callback=callback, maxiters=10)
#res1 = Optimization.solve(optprob, OptimizationOptimisers.ADAM(), maxiters=500)
res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), callback=callback, maxiters=50000)

#optprob2 = Optimization.OptimizationProblem(optf, res1.u)
#res2 = Optimization.solve(optprob2, Optim.LBFGS(), maxiters=500)
#p_trained = res2.u
#println("Final training loss after $(length(losses)) iterations: $(losses[end])")
# Rename the best candidate

#-------------------------------------------------------------------------

