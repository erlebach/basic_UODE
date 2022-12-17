# Follow demo at https://docs.sciml.ai/Overview/dev/showcase/missing_physics/

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
##-----------------------------------------------------------------------
# Generating the Training Data
# First, let's generate training data from the Lotka-Volterra equations. This is straightforward and standard DifferentialEquations.jl usage. Our sample data is thus generated as follows:

function lotka!(du, u, p, t)
  α, β, γ, δ = p
  du[1] = α*u[1] - β*u[2]*u[1]
  du[2] = γ*u[1]*u[2]  - δ*u[2]
end

# Define the experimental parameter
tspan = (0.0,5.0)
u0 = 5f0 * rand(rng, 2)
p_ = [1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0,tspan, p_)
# Using tolerances of 1.e-12 not a great idea if saaving at maany times steps
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.25)

# Add noise in terms of the mean
X = Array(solution)
t = solution.t

x̄ = mean(X, dims = 2)
noise_magnitude = 5e-3
Xₙ = X .+ (noise_magnitude*x̄) .* randn(rng, eltype(X), size(X))

plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])
##-----------------------------------------------------------------------
# Definition of the Universal Differential Equation
# Now let's define our UDE. We will use Lux.jl to define the neural network as follows:
rbf(x) = exp.(-(x.^2))

# Multilayer FeedForward
U = Lux.Chain(
    Lux.Dense(2,5,rbf), Lux.Dense(5,5, rbf), Lux.Dense(5,5, rbf), Lux.Dense(5,2)
)
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)
##-----------------------------------------------------------------------
# We then define the UDE as a dynamical system that is u' = known(u) + NN(u) like:

# Define the hybrid model
function ude_dynamics!(du,u, p, t, p_true)
  û = U(u, p, st)[1] # Network prediction
  du[1] = p_true[1]*u[1] + û[1]
  du[2] = -p_true[4]*u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du,u,p,t) = ude_dynamics!(du,u,p,t,p_)
# Define the problem
prob_nn = ODEProblem(nn_dynamics!,Xₙ[:, 1], tspan, p)
println(prob_nn)
##-----------------------------------------------------------------------
#= Notice that the most important part of this is that the neural network 
   does not have hardcoded weights. The weights of the neural network are 
   the parameters of the ODE system. This means that if we change the 
   parameters of the ODE system, then we will have updated the internal 
   neural networks to new weights. Keep that in mind for the next part.

   Even if the known physics is only approximate or correct, it can be helpful to 
   improve the fitting process! Check out this JuliaCon talk which dives into this issue.
   (https://www.youtube.com/watch?v=lCDrCqqnPto)
=#
#= Setting Up the Training Loop
   Now let's build a training loop around our UDE. First, let's make a function predict 
   which runs our simulation at new neural network weights. Recall that the weights of the 
   neural network are the parameters of the ODE, so what we need to do in predict is update 
   our ODE to our new parameters and then run it.

   For this update step, we will use the remake function from the SciMLProblem interface. 
   remake works by specifying key = value pairs to update in the problem fields. 
   Thus to update u0, we would add a keyword argument u0 = .... 
   To update the parameters, we'd do p = .... The field names can be acquired from the 
   problem documentation (or the docstrings!).
=#
function predict(θ, X = Xₙ[:,1], T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                ))
end
##-----------------------------------------------------------------------
##-----------------------------------------------------------------------
#=
  Now for our loss function we solve the ODE at our new parameters and check 
  its L2 loss against the dataset. Using our predict function, this looks like:
=#
function loss(θ)
  X̂ = predict(θ)
  mean(abs2, Xₙ .- X̂)
end
##-----------------------------------------------------------------------
# Lastly, what we will need to track our optimization is to define a 
# callback as defined by the OptimizationProblem's solve interface. 
# Because our function only returns one value, the loss l, the callback 
# will be a function of the current parameters θ and l. Let's setup a 
# callback prints every 50 steps the current loss:
losses = Float64[]

callback = function (p, l)
  push!(losses, l)
  if length(losses)%50==0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
  end
  return false
end
##-----------------------------------------------------------------------
#= 
   Training
Now we're ready to train! To run the training process, we will need to 
build an OptimizationProblem. Because we have a lot of parameters, we will 
use Zygote.jl. Optimization.jl makes the choice of automatic diffeerentiation 
easy just by specifying an adtype in the OptimizationFunction construction
=#
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
##-----------------------------------------------------------------------
#=
  Now... we optimize it. We will use a mixed strategy. First, let's do some 
  iterations of ADAM because it's better at finding a good general area of 
  parameter space, but then we will move to BFGS which will quickly hone in 
  on a local minima. Note that if we only use ADAM it will take a ton of 
  iterations, and if we only use BFGS we normally end up in a bad local minima, 
  so this combination tends to be a good one for UDEs.

  Thus we first solve the optimization problem with ADAM. Choosing a learning 
  rate of 0.1 (tuned to be as high as possible that doesn't tend to make the 
  loss shoot up), we see:
=#
##-----------------------------------------------------------------------
res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters = 5000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
##-----------------------------------------------------------------------
#=
  Now we use the optimization result of the first run as the initial condition 
of the second optimization, and run it with BFGS. This looks like:
=#
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters = 1000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Rename the best candidate
p_trained = res2.u

# We now have a trained UDE

# BFGS converged much lower than the previous demo. Why? HOw sensitive are these 
# results to parameter choices? 
##-----------------------------------------------------------------------
# Plot the losses
pl_losses = plot(1:5000, losses[1:5000], yaxis = :log10, xaxis = :log10, 
    xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(5001:length(losses), losses[5001:end], yaxis = :log10, xaxis = :log10, 
    xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
##-----------------------------------------------------------------------
#=
   Next, we compare the original data to the output of the UDE predictor. 
   Note that we can even create more samples from the underlying model by simply adjusting the time steps!
=#
## Analysis of the trained network
# Plot the data and the approximation
ts = first(solution.t):mean(diff(solution.t))/2:last(solution.t)
X̂ = predict(p_trained, Xₙ[:,1], ts)
# Trained on noisy data vs real solution
pl_trajectory = plot(ts, transpose(X̂), xlabel = "t", ylabel ="x(t), y(t)", color = :red, label = ["UDE Approximation" nothing])
scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Measurements" nothing])
##-----------------------------------------------------------------------
# Lets see how well the unknown term has been approximated:
# Ideal unknown interactions of the predictor
Ȳ = [-p_[2]*(X̂[1,:].*X̂[2,:])';p_[3]*(X̂[1,:].*X̂[2,:])']
# Neural network guess
Ŷ = U(X̂,p_trained,st)[1]

pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["UDE Approximation" nothing])
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
##-----------------------------------------------------------------------
# And have a nice look at all the information:
# Plot the error
pl_reconstruction_error = plot(ts, norm.(eachcol(Ȳ-Ŷ)), yaxis = :log, xlabel = "t", ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2,1))

pl_overall = plot(pl_trajectory, pl_missing)
##-----------------------------------------------------------------------
# That looks pretty good. And if we are happy with deep learning, we can leave it 
#  at that: we have trained a neural network to capture our missing dynamics.
##-----------------------------------------------------------------------
#=
  Symbolic regression via sparse regression (SINDy based)
  Okay that was a quick break, and that's good because this next part is pretty cool. 
  Let's use DataDrivenDiffEq.jl to transform our trained neural network from machine 
  learning mumbo jumbo into predictions of missing mechanistic equations. 
  To do this, we first generate a symbolic basis that represents the space of mechanistic functions we believe this neural network should map to. Let's choose a bunch of polynomial functions
=#
##-----------------------------------------------------------------------
@variables u[1:2]
b = polynomial_basis(u, 4)
basis = Basis(b,u);
##-----------------------------------------------------------------------
#=
  Now let's define our DataDrivenProblems for the sparse regressions. To assess the 
  capability of the sparse regression, we will look at 3 cases:

  1) What if we trained no neural network and tried to automatically uncover the 
     equations from the original noisy data? This is the approach in the literature 
     known as structural identification of dynamical systems (SINDy). We will call 
     this the full problem. This will assess whether this incorporation of prior 
     information was helpful.
  
  2)  What if we trained the neural network using the ideal right hand side missing 
      derivative functions? This is the value computed in the plots above as Ȳ. 
      This will tell us whether the symbolic discovery could work in ideal situations.
  
  3)  Do the symbolic regression directly on the function y = NN(x), i.e. the trained 
      learned neural network. This is what we really want, and will tell us how to 
      extend our known equations.

  To define the full problem, we need to define a DataDrivenProblem that has the 
  time series of the solution X, the time points of the solution t, and the derivative 
  at each time point of the solution (obtained by the ODE solution's interpolation. 
  We can just use an interpolation to get the derivative:
=#
# Took a long time to execute the next line
full_problem = ContinuousDataDrivenProblem(Xₙ, t)
ideal_problem = DirectDataDrivenProblem(X̂, Ȳ)
nn_problem = DirectDataDrivenProblem(X̂, Ŷ)
##-----------------------------------------------------------------------
λ = exp10.(-3:0.01:3)
opt = ADMM(λ)
# This is one of many methods for sparse regression, consult the 
#  DataDrivenDiffEq.jl documentation for more information on the algorithm 
# choices. Taking this, let's solve each of the sparse regressions:
options = DataDrivenCommonOptions(
    maxiters = 10_000, normalize = DataNormalization(ZScoreTransform), selector = bic, digits = 1,
    data_processing = DataProcessing(split = 0.9, batchsize = 30, shuffle = true, rng = StableRNG(1111)))

full_res = solve(full_problem, basis, opt, options = options)
full_eqs = get_basis(full_res)
println(full_res)
println(full_res.residuals)
# Residuals: 197.8, Rackauckas:  197.9 (same to 5-6 digits)
##-----------------------------------------------------------------------
options = DataDrivenCommonOptions(
    maxiters = 10_000, normalize = DataNormalization(ZScoreTransform), selector = bic, digits = 1,
    data_processing = DataProcessing(split = 0.9, batchsize = 30, shuffle = true, rng = StableRNG(1111)))

ideal_res = solve(ideal_problem, basis, opt, options = options)
ideal_eqs = get_basis(ideal_res)
println(ideal_res)
# Residuals: 1.e-30 (Rackauckas: 6.1)
##-----------------------------------------------------------------------
options = DataDrivenCommonOptions(
    maxiters = 10_000, normalize = DataNormalization(ZScoreTransform), selector = bic, digits = 1,
    data_processing = DataProcessing(split = 0.9, batchsize = 30, shuffle = true, rng = StableRNG(1111)))

nn_res = solve(nn_problem, basis, opt, options = options)
nn_eqs = get_basis(nn_res)
println(nn_res)
# Residuals: 12.12, Rackauckas: 12.23
##-----------------------------------------------------------------------
#=
  Note that we passed the identical options into each of the solve calls to 
  get the same data for each call.

  We already saw that the full problem has failed to identify the correct 
  equations of motion. To have a closer look, we can inspect the corresponding equations:
=#
for eqs in (full_eqs, ideal_eqs, nn_eqs)
  println(eqs)
  println(get_parameter_map(eqs))
  println()
end
##-----------------------------------------------------------------------
# Next, we want to predict with our model. To do so, we embedd the basis into 
# a function like before:

# Define the recovered, hybrid model
function recovered_dynamics!(du,u, p, t)
  û = nn_eqs(u, p) # Recovered equations
  du[1] = p_[1]*u[1] + û[1]
  du[2] = -p_[4]*u[2] + û[2]
end

estimation_prob = ODEProblem(recovered_dynamics!, u0, tspan, get_parameter_values(nn_eqs))
estimate = solve(estimation_prob, Tsit5(), saveat = solution.t)

# Plot
plot(solution)
plot!(estimate)
##-----------------------------------------------------------------------
# We are still a bit off, so we fine tune the parameters by simply 
#  minimizing the residuals between the UDE predictor and our 
#  recovered parametrized equations:

function parameter_loss(p)
  Y = reduce(hcat, map(Base.Fix2(nn_eqs, p), eachcol(X̂)))
  sum(abs2, Ŷ .- Y)
end

optf = Optimization.OptimizationFunction((x,p)->parameter_loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))
parameter_res = Optimization.solve(optprob, Optim.LBFGS(), maxiters = 1000)
##-----------------------------------------------------------------------
# Simulation
# Look at long term prediction
t_long = (0.0, 50.0)
estimation_prob = ODEProblem(recovered_dynamics!, u0, t_long, parameter_res)
estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.1) # Using higher tolerances here results in exit of julia
plot(estimate_long)
##-----------------------------------------------------------------------
true_prob = ODEProblem(lotka!, u0, t_long, p_)
true_solution_long = solve(true_prob, Tsit5(), saveat = estimate_long.t)
plot!(true_solution_long)
##-----------------------------------------------------------------------
# Post Processing and Plots
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
##-----------------------------------------------------------------------
##-----------------------------------------------------------------------
##-----------------------------------------------------------------------
##-----------------------------------------------------------------------
##-----------------------------------------------------------------------
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
sampler = DataProcessing(split = 0.8, shuffle = true, batchsize = 30, rng = rng)
nn_res = solve(nn_problem, basis, opt, maxiter=10, progress=true, data_processing=sampler, digits=1)

# Store the results
results = [full_res; ideal_res; nn_res]
#------------------------------------------------------------------------
# Show the results
map(println, results)
# Show the results  (??? result not defined)
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

estimation_prob = ODEProblem(recovered_dynamics!, u0, tspan, parameters(nn_res))
estimate = solve(estimation_prob, Tsit5(), saveat = solution.t)

# Plot
plot(solution)
plot!(estimate)
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
