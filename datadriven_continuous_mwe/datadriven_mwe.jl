using ModelingToolkit, DataDrivenDiffEq, DataDrivenSparse

X = Float64.(randn(2, 31))
Y = X .^2 .+ 3 .* X .+ 5
DX = rand(2, 31)
t = 0:.1:3

@variables u[1:2]
b = polynomial_basis(u, 5)
push!(b, sin.(u[1]))
push!(b, sin.(u[2]))
basis = Basis(b, u);

λ = exp10.(-3:0.01:5)
opt = STLSQ(λ)

full_problem = DataDrivenProblem(X, t=t, DX = DX)
res = solve(full_problem, basis, opt, maxiter=100, progress=true)

function subtypetree(t, level=1, indent=4)
    level == 1 && println(t)
    for s in subtypes(t)
      println(join(fill(" ", level * indent)) * string(s))
      subtypetree(s, level+1, indent)
    end
end