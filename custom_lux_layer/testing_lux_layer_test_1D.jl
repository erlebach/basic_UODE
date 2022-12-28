# 2022-12-25
# Get the poly layer working on a simple example, and then perhaps integeate this into 
# standalone_lux_layer.jl
#
using Revise
using Zygote
using ForwardDiff
using Lux
using Random
using Statistics
using Tullio
using Plots
using Optimisers
using ComponentArrays
using LinearAlgebra

include("./Polynomial.jl")
include("./Polynomial_layer.jl")

# ================== END DEFINITION of Polynomial Layer =======================================

# Let us define a quadratic functions and apply the polynomial layer 
# to train the parameters

rng = Random.default_rng()
Random.seed!(rng, 0)

# I set up the problem incorrectly! Teh polynomials shoudl be functions of (u1, u2), where 
# u1, u2 are the input to the neural network. 
function generate_data_1D(N)
    x = range(-2.f0, 2.f0, N) |> collect 
    p = (0, 1, 2)    # poly(x) = x + x^2

    # To avoid broadcase of p, one must wrap p in a container as below
    # https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting
    #y = evalpoly.(x, (p,))  # .+ randn(rng, (1, 128)) * .1f0
    y = evalpoly.(x, Ref(p)) .+ randn(rng, N) * .1f0
    println("==> size y: ", size(y))
    return x, y
end

# Generate data
N = 128
degree = 3 
x, y = generate_data_1D(N)
data = y
println("testing: size x, y: $(size(x)), $(size(y))")

# Generate polynomial function
# is y[1,:] equivalent to a view?
# next two lines WORK here!
#@time poly, coef = generate_polynomial(N, degree, y[1,:], y[2,:]);
#poly(rand(length(coef))) # evaluate the polynomial at coef
# For now, the poly itself is generated within the poly layer. But the generated polynomial should be in 
# an initializer. So I'd like to imoplement initializer functionality. 
plot(x, y)

model = Polylayer(; out_dims=1, degree=3, init_weight=Lux.rand32)
ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)
opt = Lux.Adam(0.01f0)
# My cost function is not decreasing. Something is clearly wrong. 
st_opt = Optimisers.setup(opt, ps)
dct = Dict(:rng => rng, :model => model, :ps => ps, :st => st, :opt => opt, :st_opt => st_opt)

function loss_function(model, ps, st, data)
    y_pred, _ = model(data, ps, st)  # Model should return size (model.out_dims, N). Update global st
    mse_loss = mean(abs2, y_pred - data)     # mutated data?
    # induce sparsity
    lambda = 1.
    mse_loss = mse_loss + lambda * norm(ps, 1)
    println("mse_loss: ", mse_loss)
    return mse_loss, ()   # what is ()? 
end

# Finally the training loop.
function main(; model, ps, st, st_opt, data=nothing, epochs=100, kwargs...)
    # needed because st_opt on left-hand side further down, and a 
    # variable can only have a single declaration in a function
    data = rand(128, model.out_dims)  # shadows the argument

    # Next: need a call function. Perhaps use Optimization.jl package? 
    
    gs = 0
    for epoch in 1:epochs
        gs = Zygote.gradient((coef, data) -> loss_function(model, coef, st, data)[1], ps, data)[1]
        st_opt, ps = Optimisers.update(st_opt, ps, gs)  # ERROR
    end
    return ps, gs
end

# I wasted three days on this. And it does not appear to work. Why not? 

new_ps, gs = main(; dct..., data=data, epochs=1000)
ps.coeffs
new_ps.coeffs
println("original _ps: ", ps.coeffs)
println("new_ps: ", new_ps)
size(new_ps)

# the data is x = x + x^2. Therefore the correct coefficients should be: 
"""
I = (0, 1, 0, 2, 1, 0, 3, 2, 1, 0)
J = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3)
"""

# The functions are approx the same, so should the coefficients. But the're not. 
# Theoretically, the nonzero coefficients should be indexes: 
#   for the x equation: index 2 and 4 
#   for the y equation: index 3 and 5 (since the equations are decoupled)