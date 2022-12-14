using ModelingToolkit  # for @variables
using DataDrivenDiffEq
using DataDrivenSparse

# Given a function:
#    g(u,v) = sin(1.7*x) + u^3 + 1.5*u
#
#  find a sparse basis representation
# From https://docs.sciml.ai/DataDrivenDiffEq/dev/libs/datadrivensparse/examples/example_01/  
# dev version

# This problem is Y fitted with X
# According to Chris Rackauckas, this functionality should be a part of the Module in a few days (2022-12-11). 
# At least, Chris stated that an estimate of the parameters should be returned. To revisit. Currently, code crasshes. 

function setupFunction(;n=100)
    g(u) = u .^2 .+ 2.5 .* u.^4 .- 1
    h(u) = sin.(u) + u.^3 .+ 1.5 .* u
    X = randn(n)
    Y = g(X)
    Z = h(X)
    # X,Y must have at least two columns
    X = reshape(X, 1, n)
    Y = reshape(Y, 1, n)
    Z = reshape(Z, 1, n)
    Y = vcat(Y,Z)
    return X, Y
end

function sparseBasis(X, Y; n=100)
    problem = DirectDataDrivenProblem(X, Y, name=:test)
    @parameters q[1]
    @variables u[1]
    basis = monomial_basis(u, 7)
    push!(basis, sin.(q * u)) 
    basis = Basis(basis, u, parameters=q)
    # STLQ is one of several possible algorithms
    res = solve(problem, basis, STLSQ())

    # What is the fit?
    println(res |> get_basis)
    println("Parameters: ", res |> get_basis |> get_parameter_values)
    println("residual: ", res.residuals)
end

X, Y = setupFunction(;n)
sparseBasis(X, Y; n=100)