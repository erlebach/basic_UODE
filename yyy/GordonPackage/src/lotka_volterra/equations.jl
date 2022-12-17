
function lotka!(du, u, p, t)
  α, β, γ, δ = p
  du[1] = α*u[1] - β*u[2]*u[1]
  du[2] = γ*u[1]*u[2]  - δ*u[2]
end

function setupLotka!(dict, system_eq)
	prob = ODEProblem(system_eq!, dict[:u0],dict[:tspan], dict[:p_])
	# Using tolerances of 1.e-12 not a great idea if saaving at maany times steps
	solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.25)

	# Add noise in terms of the mean
	X = Array(solution)
	t = solution.t
	x̄ = mean(X, dims = 2)
	noise_magnitude = 5e-3
	Xₙ = X .+ (noise_magnitude*x̄) .* randn(rng, eltype(X), size(X))
	return solution, Xₙ

