using Revise
import GordonPackage as G
import Parameters as P
import Dictionaries as D

#----------------------------------
# Packages in GordonPackage (should not be required)
using StableRNGs
# using OrdinaryDiffEq
# using Statistics
# using Plots
# using Lux
# using Optimization
# using ComponentArrays
# using Zygote
# using DiffEqFlux
# using LinearAlgebra
#----------------------------------
###
### # Set a random seed for reproducible behaviour
rng = StableRNG(1111)
eq_to_solve = G.lotka!

# I have not decided whether to work with NamedTupples or Dictionaries

# Access via dict["tspan"], dict[:u0].
# Mutable
dct = Dict{Symbol, Any}(
        :u0 => 5f0 * rand(rng,2),
        :tspan => (0.0, 5.0),
        :p_ => [1.3, 0.9, 0.8, 1.8],
        :RNG => rng
    )

println("dict: ", dct)

solution, Xₙ =  G.setupLotka(dct, eq_to_solve)
t = solution.t
# I could set these inside G.setupLotka, but then I lose track of when the dct elements are set, complicating debugging. 
dct[:solution] = solution
dct[:Xₙ] = Xₙ
dct[:t] = t

rbf(x) = exp.(-(x.^2))

# Using dictionaries allows me to leave function arguments untouched. 
# Optimziations can occur later on if necessary
NN_dct = Dict{Symbol, Any}(
    :layer_sz => 5,
    :nb_layers => 2,
    :activation => rbf,
    :iter_ADAM => 7501,  # 5001
    :iter_LBFGS => 1001    # 1001
)

# Get the initial parameters and state variables of the model
#NN = G.NN_model(; NN_dct...)
u = (.1, .2)
NN = G.tbnn(u; NN_dct...)
#NN = G.NN_model_sindy(u; NN_dct...)
p, st = Lux.setup(rng, NN)

# Add additional elements to the NN dictionary
NN_dct[:NN] = NN
NN_dct[:p] = p
NN_dct[:st] = st
##----------------------------------------------------------------------
# Put everything needed to solve the UODE into a single function. To solve another 
# problem, simply write a new function with different content. 
# To experiment with different parameters, change parameter values in dct and NN_dct
NN_dct[:losses] = Float64[]
res1, res2 = G.solve_UODE(dct, NN_dct)
p_adam_trained = res1.u
p_bgfs_trained = res2.u

# Plot the losses
iter = NN_dct[:iter_ADAM]
losses = NN_dct[:losses]
pl_losses = plot(1:iter, losses[1:iter], yaxis=:log10, xaxis=:log10,
    xlabel="Iterations", ylabel="Loss", label="ADAM", color=:blue)
pl_losses
iter1 = NN_dct[:iter_LBFGS]
plot!(iter+1:length(losses), losses[iter+1:end], yaxis=:log10, xaxis=:log10,
    xlabel="Iterations", ylabel="Loss", label="BFGS", color=:red)
##-----------------------------------------------------------------------
# Trained on noisy data vs real solution
X̂ = dct[:X̂]
l_trajectory = plot(ts |> collect, transpose(X̂), xlabel = "t", ylabel ="x(t), y(t)", color = :red, label = ["UDE Approximation" nothing])
scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Measurements" nothing])
##-----------------------------------------------------------------------
# Lets see how well the unknown term has been approximated:
# Ideal unknown interactions of the predictor
p_ = dct[:p_]
Ȳ = [-p_[2]*(X̂[1,:].*X̂[2,:])';p_[3]*(X̂[1,:].*X̂[2,:])']
# Neural network guess
Ŷ = NN(X̂, p_trained, st)[1] # Why the 1st index?
print("size(Ŷ): ", size(Ŷ))

pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["UDE Approximation" nothing])
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
##-----------------------------------------------------------------------
#=
   Next, we compare the original data to the output of the UDE predictor.
   Note that we can even create more samples from the underlying model by simply adjusting the time steps!
=#
## Analysis of the trained network
# Plot the data and the approximation
ts = first(solution.t):mean(diff(solution.t))/2:last(solution.t)
# Notice that I captured an interior function. 
prob_nn = ODEProblem(NN_dct[:nn_dynamics!], dct[:Xₙ][:, 1], dct[:tspan], NN_dct[:p])
p_trained = p_bgfs_trained
# I only have to add the keywords of the parameters different than the ODEProblem
X̂ = NN_dct[:predict](p_trained; X=dct[:Xₙ][:,1], T=ts)

# Trained on noisy data vs real solution
pl_trajectory = plot(ts, transpose(X̂), xlabel = "t", ylabel ="x(t), y(t)", color = :red, label = ["UDE Approximation" nothing])
scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Measurements" nothing])
##-----------------------------------------------------------------------
# Lets see how well the unknown term has been approximated:
# Ideal unknown interactions of the predictor
p_ = dct[:p_]
Ȳ = [-p_[2]*(X̂[1,:].*X̂[2,:])';p_[3]*(X̂[1,:].*X̂[2,:])']
# Neural network guess
U = NN_dct[:NN]
Ŷ = U(X̂,p_trained,st)[1]

pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["UDE Approximation" nothing])
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
##-----------------------------------------------------------------------
# And have a nice look at all the information:
# Plot the error
pl_reconstructilabel
##-----------------------------------------------------------------------