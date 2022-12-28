
using Revise
using Lux
using Random
using Statistics
using Plots
using Tullio

include("./Polynomial.jl")

"""
Define a custom Polynomial layer in Lux.jl to allow the polynomial coefficients to 
be estimated through training. 
"""

struct Polylayer{F} <: Lux.AbstractExplicitLayer
    out_dims::Int
    degree::Int
    init_weight::F
end

function Base.show(io::IO, d::Polylayer)
    println(io, "Polylayer(out_dims: $(d.out_dims)), degree: $(d.degree)")
end

function Polylayer(out_dims::Int, degree::Int, init_weight=Lux.rand32)
    dtypes = (typeof(init_weight))
    return Polylayer{dtypes}(out_dims, degree, init_weight)
end

# Allow arbitrary order for arguments with default values
function Polylayer(; out_dims::Int=2, degree::Int=2, init_weight=Lux.rand32)
    dtypes = (typeof(init_weight))
    return Polylayer{dtypes}(out_dims, degree, init_weight)
end

"""
Evaluate the polynomial whose coefficients are given by ps.coeffs (ps.coeffs[1] is lowest degree coefficient)
ps.coeffs is a matrix of size [:, degree]

What to do when there are batches of points? 
"""
function (l::Polylayer)(x::AbstractMatrix, ps, st::NamedTuple)
    c = ps.coeffs
    x1 = reshape(x, l.out_dims, :) # lasts dimension is the number of training samples
    println("size(x1): ", size(x1))

    N = size(x1, length(size(x1)))
    sum = zeros(l.out_dims, N)

    # ARE THERE BETTER APPROACHES WITH BROADCAST?

    # Evaluate polynomial
    # for j in 1:l.out_dims
    for i in 1:N
        sum[:, i] = c[:, end]
    end
    """
    for d in l.degree : -1 : 1
        for i in 1:N
            sum[:, i] .= sum[:, i] .* x1[:, i]  .+ c[:, d]
        end
    end
    """
    println("out of degree d loop, size(sum): ", size(sum))
    return sum, st
end


# overload initialparameters in the Lux environment
# coeffs[out_dims, degree+1]
function Lux.initialparameters(rng::AbstractRNG, d::Polylayer)
     # why the final comma?
     return (coeffs=d.init_weight(rng, out_dims, d.degree+1),)
end

# overload initialstates in the Lux environment
Lux.initialstates(::AbstractRNG, ::Polylayer) = NamedTuple()

statelength(d::Polylayer) = 0

function parameterlength(d::Polylayer)
    return d.degree + 1
end

# ================== END DEFINITION =======================================

function test_layer()
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    out_dims = 2
    degree = 3
    model = Polylayer(; out_dims=out_dims, degree=degree, init_weight=Lux.rand32)

    ps, st = Lux.setup(rng, model)
    initialparameters(rng, model)

    println(model)
    Lux.ComponentArray(ps)

    x = rand(out_dims)

    # Evaluate the layer
    y = model(x', ps, st)[1]
end

# Let us define a quadratic functions and apply the polynomial layer 
# to train the parameters

function generate_data()
    x = range(-2.f0, 2.f0, 128) |> collect 
    x = reshape(x, (1, 128))
    p = (0, 1, 2)  

    # To avoid broadcase of p, one must wrap p in a container as below
    # https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting
    #y = evalpoly.(x, (p,))  # .+ randn(rng, (1, 128)) * .1f0
    y1 = evalpoly.(x, Ref(p)) .+ randn(rng, (1, 128)) * .1f0
    y2 = evalpoly.(x, Ref(p)) .+ randn(rng, (1, 128)) * .1f0
    y = vcat(y1, y2)
    return x, y
end

x, y = generate_data()
plot(reshape(x, :), y[1,:])
plot!(reshape(x, :), y[2,:])

model = Polylayer(; out_dims=2, degree=3, init_weight=Lux.rand32)
ps, st = Lux.setup(rng, model)

opt = Lux.Adam(0.03f0)

function loss_function(model, ps, st, data)
    # in original code, feelding the x into the network to predict y
    # in our case, we feed 128 pairs (u1,u2)
   # y_pred, st = Lux.apply(model, data[1], ps, st) # orig
    y_pred, st = Lux.apply(model, data[2]', ps, st)
    mse_loss = mean(abs2, y_pred .- data[2])
    return mse_loss, st, ()
end

# Now we will use Zygote for our AD requirements.
vjp_rule = Lux.Training.ZygoteVJP()


# First we will create a [`Lux.Training.TrainState`](@ref) which is essentially a
# convenience wrapper over parameters, states and optimizer states.
tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=cpu)

# Finally the training loop.
function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data::Tuple, epochs::Int)
    #data = data .|> gpu
    # no batches? 
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function, data, tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end

tstate = main(tstate, vjp_rule, (x, y), 250)
#y_pred = cpu(Lux.apply(tstate.model, Lux.gpu(x), tstate.parameters, tstate.states)[1])
#y_pred = Lux.apply(tstate.model, x, tstate.parameters, tstate.states)[1]
