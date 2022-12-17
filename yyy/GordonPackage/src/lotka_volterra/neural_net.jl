
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

#---------------------------------------------
# Define the Neural Netowrk model
#α, β, γ, δ = p  # accessing named tuple (WHY?)

rbf(x) = exp.(-(x.^2))
act = tanh
layer_size = 10 

dudt2 = model = construct_model(layer_size, rbf)
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, model)

#Define the hybrid model
# Missing

neuralODEfunc(du, u, p, t) = lotka_volterra_NN!(du, u, p, t, p_LV)

#define the problem with noisy solution as initial conditions, and NN as some of its terms
prob_nn = ODEProblem(neuralODEfunc, Xₙ[:, 1], tspan, p)

#------------------------------------------
# Train the network
adtype = Optimization.AutoZygote()
# adtype = Optimization.AutoFiniteDiff()
# ERROR: Cannot find function signature. Must wait on Rackauckas
optf = Optimization.OptimizationFunction((x,p)->loss_neuralode(x), adtype)

optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
res1 = Optimization.solve(optprob, ADAM(0.1), callback=callback, maxiters=200)

println("Training loss initially: $(losses[1])")
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.minimizer)
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
# The reconstruction is really poor
pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["UDE Approximation" nothing])
# ideal
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
#-----------------------------------------------------------------------
# Plot the error
pl_reconstruction_error = plot(ts, norm.(eachcol(Ȳ-Ŷ)), yaxis = :log, xlabel = "t", ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2,1))
#-----------------------------------------------------------------------
