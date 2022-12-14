using ModelingToolkit  # for @variables
using DataDrivenDiffEq
using DataDrivenSparse

# Given function f(u,v) = -1 + 2.5*u^3 + u*v + v^2
# Generate u, v, f(u,v)

# fit f
# From https://docs.sciml.ai/DataDrivenDiffEq/dev/libs/datadrivensparse/examples/example_01/  
# dev version

# This problem is Y fitted with X

function setupFunction(n=100)
    g(u,v) = -1 .+ (2.5 .* u.^3) .+ (u .* v) .+ (v .^ 2)
    # h(u,v) = g(u,v)  # The two functions can be the same
    h(u,v) = (2.5 .* v.^3) .+ (u .* v) .+ (u .^ 2)
    X = randn(n)
    Y = randn(n)
    Z1 = reshape(g(X,Y), 1, n)
    Z2 = reshape(h(X,Y), 1, n)
    Z = vcat(Z1, Z2)
    # X,Y must have at least two columns
    X = reshape(X, 1, n)
    Y = reshape(Y, 1, n)
    X = vcat(X, Y)
    return X, Z
end

function sparseBasis(X, Y; n=100)
    problem = DirectDataDrivenProblem(X, Y, name=:test)
    @variables u[1:2]
    basis = Basis(polynomial_basis(u, 5), u)
    println(basis)
    println("methods: ", @which solve(problem, basis, STLSQ()))
    res = solve(problem, basis, STLSQ())

    # What is the fit?
    println(res |> get_basis)
    println(res |> get_basis |> get_parameter_values)
    println("residual: ", res.residuals)
    println(size(X), size(Y))
    println("type(X, Y): ", typeof(X), typeof(Y))
end

X, Y = setupFunction(n)
sparseBasis(X, Y; n=100)