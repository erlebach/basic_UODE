# 2022-12-25
# 1D network that outputs the coefficients of a polynomial
#
# The network outputs the coefficients of the polynomial approximation
# Given: $x_i$, $i=1,\cdots,N$, that satisfy a polynomial of degree $d$. 
# The objective is to compute the nonzero coefficients of this polynomial, 
# coupled with the constraint that the output should be independent of $x_i4. 
# One way is to 

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

sz = 3
rng = Random.default_rng()
Random.seed!(rng, 0)

N = 128
nb_terms = 4
model = Chain(Dense(1, sz, tanh), 
            Dense(sz, sz, tanh), 
            Dense(sz, nb_terms))
Lux.initialparameters(rng, model)
Lux.initialstates(rng, model)
ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)

# Feed x into the network, generate Coef

# transform matrix from sizeshape (N) to shape (1,N)
x = reshape(rand(N), 1, N)
y, st = model(x, ps, st)

function loss_function(model, x_data, y_data, ps, st)  # st not used
    coef_pred, st = model(x_data, ps, st)
    # Evaluate the predicted polynomial (degree 3)
    # y_data: (128,)
    # x_data: (128,)
    # coef_pred: (4, 128)  # Should be 4
    println("x_data: ", size(x_data))
    println("y_data: ", size(y_data))
    println("coef_pred: ", size(coef_pred)) # 
    x1 = reshape(x_data, :)
    x2 = reshape(x_data .^ 2, :)
    x3 = reshape(x_data .^ 3, :)
    pred_poly = coef_pred[1] .+ coef_pred[2] .* x1 .+ coef_pred[3] .* x2 .+ coef_pred[4] .* x3
    println("pred_poly: ", size(pred_poly))
    loss = mean( (y_data .-pred_poly) .^ 2)
    println("loss: ", loss)
    # How does one sparsify the output?  L1-norm on the output (pred_poly)? 
    return loss
end

function train(model, ps, st)
    N = 128
    epochs = 100
    opt = Adam(0.01)
    st_opt = Optimisers.setup(opt, ps)
    x_data = rand(N)
    y_data = rand(N)
    for e in 1:epochs
        println("epoch: ", e)
        gs = gradient(x -> loss_function(model, x, y_data, ps, st), reshape(x_data, 1, :))
        println("gs: ", gs) # ok
        st_opt, ps = Optimisers.update(st_opt, ps, gs) # ERROR
    end
end

# ERROR in train), on line 57, Optimisers.update
train(model, ps, st)

gs = gradient(x -> loss_function(model, x, y_data, ps, st), reshape(x_data, 1, :))



x_data = rand(N)
y_data = rand(N)
ps
st

# transform matrix from sizeshape (N) to shape (1,N)
model(reshape(x_data, 1, :), ps, st)

# 0.2 sec (very dlow)
function test()
    @time res = gradient(x -> loss_function(model, x, y_data, ps, st), reshape(x_data, 1, :))
    return res
end

res = test()
println(res)