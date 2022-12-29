# 2022-12-25
"""
Solve a 1D equation of one variable of the form
  dy/dt = -y + .1 y^2 - .3 y^4
using UODE in the form: 
  dy/dt = -y + NN(y)
where NN is a neural network with one input and one output.
Starting point: testing_lux_layer_test_1D.jl
File name: 1D_equation_with_poly_layer_1D.jl

This will be followed by a two ODEs in two variables. 
"""

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

# using Optimization
# using OptimizationOptimisers
# using OptimizationOptimJL

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
    return x, y
end

# Generate data
N = 128
degree = 3 
x, y = generate_data_1D(N)
x_data = x
y_data = y
println("testing: size x, y: $(size(x)), $(size(y))")

# Generate polynomial function
# is y[1,:] equivalent to a view?
# next two lines WORK here!
#@time poly, coef = generate_polynomial(N, degree, y[1,:], y[2,:]);
#poly(rand(length(coef))) # evaluate the polynomial at coef
# For now, the poly itself is generated within the poly layer. But the generated polynomial should be in 
# an initializer. So I'd like to imoplement initializer functionality. 
plot(x, y_data)

# initialize weights with zero. So polynomial should return zero
# If zeros32 is mispelled, Julia reverts to Lux.rand32
model = Polylayer(; out_dims=1, degree=4, init_weight=Lux.zeros32)
ps = Lux.initialparameters(rng, model)
#ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)
opt = Lux.Adam(0.1f0)
# My cost function is not decreasing. Something is clearly wrong. 
st_opt = Optimisers.setup(opt, ps)
N = 128
dct = Dict(:rng => rng, :model => model, :ps => ps, :st => st, 
           :opt => opt, :st_opt => st_opt, :N => N, 
           :x_data => x, :y_data => y_data)

# x must be a 2D vector (do not know why)
# y returned is a 1D vector
y_pred = model(x, ps, st)[1]
plot(x, y) # parabola
plot!(x, y_pred)  # zero

function loss_function(model, ps, st, x_data, y_data, epoch)
    y_pred, _ = model(x_data, ps, st)  # Model should return size (model.out_dims, N). Update global st
    mse_loss = mean(abs2, y_pred - data)     # mutated data?
    # induce sparsity
    lambda = .001
    mse_loss = mse_loss + lambda * norm(ps, 1)
    if (epoch % 10 == 0)
        println("mse_loss[$(epoch)]: $(mse_loss), ps: $(ps)")
    end
    return mse_loss, ()   # what is ()? 
end

# Finally the training loop.
# The semi-colon is necessary because I am using a dictionary on the calling function
# kwargs is necessary to collect the dictionary entries not present in the argument list
function main(; model, ps, st, st_opt, x_data, y_data, epochs=200, N=128, kwargs...)
    # needed because st_opt on left-hand side further down, and a 
    # variable can only have a single declaration in a function
    # Next: need a call function. Perhaps use Optimization.jl package? 
    gs = 0
    for epoch in 1:epochs
        gs = Zygote.gradient(
                (coef, data) -> loss_function(model, coef, st, x_data, y_data, epoch)[1], 
                ps, data
            )[1]
        st_opt, ps = Optimisers.update(st_opt, ps, gs)  
    end
    return ps, gs
end

# I wasted three days on this. And it does not appear to work. Why not? 

new_ps, gs = main(; dct..., data=data, epochs=500)
ps.coeffs
new_ps.coeffs
println("original _ps: ", ps.coeffs)
println("new_ps: ", new_ps)
size(new_ps)

# Plot Original polynomial against fitted polynomial
plot(x_data, y_data)
y_pred, _ = model(x_data, new_ps, st)  # Model should return size (model.out_dims, N). Update global st
plot!(x_data, y_pred)

ps_exact = Float32[0., 1., 2., 0., 0.]
y_exact = evalpoly.(x_data, Ref(ps_exact)) 
plot!(x_data, y_exact)
@show hcat(y_pred, y_exact)
