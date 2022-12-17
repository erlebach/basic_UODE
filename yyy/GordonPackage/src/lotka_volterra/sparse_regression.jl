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
