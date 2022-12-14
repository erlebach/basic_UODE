using ModelingToolkit  # for @variables
using DataDrivenDiffEq
using DataDrivenSparse

# Given function f(u,v) = -1 + 2.5*u^4 + u^2
# Generate u, v, f(u,v)

# fit f
# From https://docs.sciml.ai/DataDrivenDiffEq/dev/libs/datadrivensparse/examples/example_01/  
# dev version

# This problem is Y fitted with X

function setupFunction(n=100)
    g(u) = u .^2 .+ 2.5 .* u.^4 .- 1
    n = 100
    X = randn(n)
    Y = g(X)
    # X,Y must have at least two columns
    X = reshape(X, 1, n)
    Y = reshape(Y, 1, n)
    return X, Y
end

function sparseBasis(X, Y; n=100)
    problem = DirectDataDrivenProblem(X, Y, name=:test)
    @variables u[1]
    basis = Basis(monomial_basis(u, 7), u)
    res = solve(problem, basis, STLSQ())

    # What is the fit?
    println(res |> get_basis)
    println(res |> get_basis |> get_parameter_values)
    print("residual: ", res.residuals)
end

n = 100
X, Y = setupFunction(n)
sparseBasis(X, Y; n=100)