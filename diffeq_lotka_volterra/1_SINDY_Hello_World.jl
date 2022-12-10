using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
using Plots
using Zygote
##

f(u) = u.^2 .+ 2.0u .- 1.0
x = randn(1, 100);
y = f(x)
# y = reduce(hcat, map(f, eachcol(x)));
scatter(x',y')

## Define basis
@variables u
basis = Basis(monomial_basis([u], 2), [u])

##
problem = DirectDataDrivenProblem(x, y)
sol = solve(problem, basis, STLSQ())
print("Done!")
##
println(sol)
println(result(sol))
params = println(sol.parameters)

# BUT HOW TO EVALUATE BACK FROM THE BASIS AND THE PARAMETERS?

struct Linear
    W
    b
end

(l::Linear)(x) = l.W * x .+ l.b
model = Linear(rand(2,5), rand(2))
model(x)
x
dmodel = gradient(model->sum(model(x)), model)[1]
model.W |> size
size(x)
x = rand(5)
dmodel = gradient(model->sum(model(x)), model)[1]
typeof(dmodel)
linear(x) = W * x .+ b
grads = gradient(()->sum(linear(x)), Params([W,b])) 

function f!(x)
    x = 2 * x
    return x
end
gradient(rand(3)) do x 
    sum(f!(x))
end