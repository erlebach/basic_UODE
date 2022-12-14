using ModelingToolkit
using DataDrivenDiffEq
using DataDrivenSparse

X = Float64.(randn(2, 31))
Y = X .^2 .+ 3 .* X .+ 5
DX = rand(2, 31)
t = 0:.1:3

@variables u[1:2]
b = [polynomial_basis(u, 5); sin.(u)]
basis = Basis(b, u);

λ = exp10.(-3:0.01:5)
opt = STLSQ(λ)

full_problem = DataDrivenProblem(X, t=t, DX = DX)
solve(full_problem, basis, opt, maxiter=100, progress=true)