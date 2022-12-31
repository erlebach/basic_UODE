# 2022-12-25
# Get the poly layer working on a simple example, and then perhaps integeate this into 
# standalone_lux_layer.jl
#
# 2022-12-25
# Solve the Lotka-Volterra equations where the nonlinear term is replaced by a polynomial layer. 
# The UODE will be trained, which will provide the polynomial structure on the right-hand side.
#
# 2022-12-31
# Create a MWE to illustrate solver generating the error: 

# ERROR: Optimization algorithm not found. Either the chosen algorithm is not a valid solver
# choice for the `OptimizationProblem`, or the Optimization solver library is not loaded.
# Make sure that you have loaded an appropriate Optimization.jl solver library, for example,
# `solve(prob,Optim.BFGS())` requires `using OptimizationOptimJL` and
# `solve(prob,Adam())` requires `using OptimizationOptimisers`.

using Revise
using Zygote
using ForwardDiff
using Lux
using Random
using Statistics
using Tullio
using Plots
using Optimisers
using ComponentArrays
using LinearAlgebra
using Optimization
using OptimizationOptimJL
using DifferentialEquations

# Multilayer FeedForward
function NN_model(; nb_layers=2, layer_sz=4, activation=tanh, kwargs...)
    # kwargs absorbs additional arguments from dictionary
    #println("NN_model, kwargs: ", kwargs)
    nbl = nb_layers
    sz  = layer_sz
    act  = activation

    U = Lux.Chain(
        Lux.Dense(2, sz, act),
        Lux.Dense(sz, sz, act),  # layer 1
        Lux.Dense(sz, sz, act),  # layer 2
        Lux.Dense(sz, 2)
    )
    return U
end

# Define the hybrid model
function ude_dynamics!(du,u, p, t, p_true, st, model)
    û = model(u, p, st)[1] # Network prediction
    du[1] = p_true[1]*u[1] + û[1]
    du[2] = -p_true[4]*u[2] + û[2]
  end


  function solve_UODE(dct, NN_dct)
    # Solve a different problem by redefining NN_model and ude_dynamics and lotka!
    # The arguments should take (perhaps via dictionaries) the system of equations to solve,
    # the NN model, and the terms to model in the differential equation.
    rng = dct[:RNG]
    nn_dynamics!(du,u,p,t) = ude_dynamics!(du,u,p,t,dct[:p_], NN_dct[:st], NN_dct[:NN])
    NN_dct[:nn_dynamics!] = nn_dynamics!

    Xₙ = dct[:Xₙ]
    t  = dct[:t]
    # C.R. did not have ";"
    function predict(θ; X=Xₙ[:,1], T=t)
        NN_dct[:prob_nn] = prob_nn
        _prob = remake(prob_nn, u0=X, tspan=(T[1], T[end]), p=θ)
        # Remake replaces the original parameters of prob_nn, the time span and the I.C.
        Array(solve(_prob, Vern7(), saveat=T, abstol=1e-6, reltol=1e-6))
    end

    # Allows predict to be used from calling environment
    NN_dct[:predict] = predict

    function loss(θ, dct)
        Xₙ = dct[:Xₙ]
        t = dct[:t]
        # predict is a global wirth respect to loss
        X̂ = predict(θ; X=Xₙ[:,1], T=t)   #
        mean(abs2, Xₙ .- X̂)
    end

    # What are the arguments of callback? LOOK THIS UP.  I would rather not refer
    # to global variables

    # Adding to an existing dictionary is dangerous. Ideally, I should check that the key does not exist and throw an error
    # if it does.
    losses = NN_dct[:losses] = Float64[]

    callback = function (p, l; kwargs...)  # GE added kwargs
        push!(losses, l)
        if length(losses)%50==0
            #println("callback: kwargs: ", kwargs |> keys |> collect) # GE
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end

    # Solve problem with noise
    prob_nn = ODEProblem(nn_dynamics!, Xₙ[:, 1], dct[:tspan], NN_dct[:p])

    # Adjoint type
    p = NN_dct[:p]
    adjoint_type  = Optimization.AutoZygote()

    # predict (which solves the equation), is called in the loss function)
    optf    = Optimization.OptimizationFunction((x,p)->loss(x, dct), adjoint_type)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

    # Solve with ADAM
    res1    = Optimization.solve(optprob, ADAM(), callback=callback, maxiters = 5)
    nb_iter_adam = length(losses)
    println("Training loss after $(nb_iter_adam) ADAM iterations: $(losses[end])")

    """
    # Continue with BFGS
    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters = NN_dct[:iter_LBFGS])
    nb_iter_lbfgs = length(losses) - nb_iter_adam
    println("Final training loss after $(nb_iter_lbfgs) BFGS iterations: $(losses[end])")

    #### I would like the next section to run from outside the module. But I add this section as a check
    # Rename the best candidate
    p_trained = res2.u
    # We now have a trained UDE
    """
end
#------------------------------------------------------------------

solve_UODE(dct, NN_dct)