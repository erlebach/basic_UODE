#using OrdinaryDiffEq
using DifferentialEquations
#using ModelingToolkit  # for @variables
#using DataDrivenDiffEq
#using DiffEqSensitivity
#using  Optim
using LinearAlgebra
#using DiffEqFlux
#using Flux
#using Plots
# gr()
#using Statistics

# Set a random seed for reproduceable behaviour
#using Random
#Random.seed!(1234)

# Create a name for saving ( basically a prefix )
#svname = "Scenario_1_"

## Data generation
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
    return du
end

# Define the experimental parameter
tspan = (0.0f0,8.9f0)

u0 = Float32[0.44249296,4.6280594]
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0,tspan, p_)
solution = solve(prob, Tsit5(), abstol=1e-7, reltol=1e-7, saveat = 0.1)

# Ideal data
X = Array(solution)
t = solution.t
