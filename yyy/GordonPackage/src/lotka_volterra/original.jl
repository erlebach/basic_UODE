# Follow demo at https://docs.sciml.ai/Overview/dev/showcase/missing_physics/

##-----------------------------------------------------------------------
# Generating the Training Data
# First, let's generate training data from the Lotka-Volterra equations. This is straightforward and standard DifferentialEquations.jl usage. Our sample data is thus generated as follows:


dict = Dict()
dict[:tspan] = (0.0, 5.0)
dict[:u0] = u0 = 5f0 * rand(rng, 2)
dict[:p_] = [1.3, 0.9, 0.8, 1.8]
# Define the experimental parameter
#tspan = (0.0,5.0)
#u0 = 5f0 * rand(rng, 2)
#p_ = [1.3, 0.9, 0.8, 1.8]

solution, Xₙ = setupLotka!(dict, lotka!)
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
