# Follow demo at https://docs.sciml.ai/Overview/dev/showcase/missing_physics/
module GordonPackage
# SciML Tools
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics

# External Libraries
# Zygote required because of use of Optimization.AutoZygote in @Requires
using ComponentArrays, Lux, Zygote, Plots, StableRNGs
gr()

# Set a random seed for reproducible behaviour
rng = StableRNG(1111)
println("=======> All using ... done")
#-------------------------------------


# Use NODE to solve Lotka-Volterra equations, more specifically, the quadratic term
include("Lotka_Volterra/equations.jl")
println("=======> added equations.jl")

#include("Lotka_Volterra/model.jl")
#println("=======> added model.jl")

#include("Lotka_Volterra/neural_net.jl")
#println("=======> added neural_net.jl")

#include("Lotka_Volterra/sparse_regression.jl")
#println("=======> added regression")

end # module GordonPackage
