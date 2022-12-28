# Working version that demonstrates Zygote not working. 

using Revise
using Zygote
using ForwardDiff
using Lux
using Random
using Statistics
using Tullio
using Optimisers
using ComponentArrays
using LinearAlgebra

# Polynomial.jl
# ================== BEGIN POLYNOMIAL GENERATION ========================================
function generate_polynomial(degree)
    Ix = 0:degree
    Iy = 0:degree

    # for 1D polynomials
    IJ = [(i,j) for i in Ix for j in Iy if i + j < (degree+1)]
    # Transforming I and J to tuples does not help with the mutation issue
    I = [i[1] for i in IJ]
    J = [i[2] for i in IJ]

    # Construct a polynomial in x and y, according to the indices listed in I
    Poly(coef, x, y) = @tullio poly[i] := coef[j] * x[i]^I[j] * y[i]^J[j] grad=Dual nograd=x nograd=y nograd=I nograd=J 
    return Poly
end
# ================== END POLYNOMIAL GENERATION =======================================
"""
# The gradient is computed correctly below. So no mutations so far. 
N = 128
d = 3 
@time poly, nb_terms = generate_polynomial(d)
@time Zygote.gradient((coef, x, y) -> (sum∘poly)(coef, x, y), rand(nb_terms), rand(N), rand(N))
""";
# ================== BEGIN Polynomial Layer =======================================
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

function (l::Polylayer)(x::T, ps, st) where T <: AbstractArray{Float64} 
    #xx = reshape(x, l.out_dims, :) # last dimension is the number of training samples
    y1 = x[:,1]
    y2 = x[:,2]

	poly1 = generate_polynomial(l.degree);  # returns a function poly1(coef, x, y)
	poly2 = generate_polynomial(l.degree);

    # Return the value of the polynomials
    stack = hcat(poly1(ps.coeffs[:,1], y1, y2), poly2(ps.coeffs[:,2], y1, y2))
    return stack, st
end

# What happens if I do not prefix by Lux and I use the polylayer inside some other module. It is not clear this will work.
nb_coefs_per_poly(d::Polylayer) = Int((d.degree+1) * (d.degree+2) / 2.) 

function Lux.initialparameters(rng::AbstractRNG, d::Polylayer)
     nb = nb_coefs_per_poly(d)
     return (coeffs=d.init_weight(rng, nb, d.out_dims),) # N x out_dims
end

# overload initialstates in the Lux environment
Lux.initialstates(::AbstractRNG, ::Polylayer) = NamedTuple()
Lux.statelength(d::Polylayer) = 0

function Lux.parameterlength(d::Polylayer)
    return nb_coefs_per_poly(d) * d.out_dims
end
# ================== END DEFINITION of Polynomial Layer =======================================
# We define a quadratic functions and apply the polynomial layer 
# to train the parameters

rng = Random.default_rng()
Random.seed!(rng, 0)

function generate_data(N)
    x = range(-2.f0, 2.f0, N) |> collect 
    x = reshape(x, (N, 1))  # N x 1
    p = (0, 1, 2)  

    # To avoid broadcase of p, one must wrap p in a container as below
    # https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting
    #y = evalpoly.(x, (p,))  # .+ randn(rng, (1, 128)) * .1f0
    y1 = evalpoly.(x, Ref(p)) .+ randn(rng, (N, 1)) * .1f0
    y2 = evalpoly.(x, Ref(p)) .+ randn(rng, (N, 1)) * .1f0
    print(size(y1))
    y = hcat(y1, y2)
    return x, y
end

# Generate data
N = 128
degree =4 
x, y = generate_data(N)

model = Polylayer(; out_dims=2, degree=4, init_weight=Lux.rand32)
ps, st = Lux.setup(rng, model)

# ComponentArray is a projector: ComponentArray(ComponentArray(ps)) === ComponentArray(ps)
# Applying it multiple times in error is not harmful
ps = ComponentArray(ps)
opt = Lux.Adam(0.03f0)

# I do not quite understand the leaf system
st_opt = Optimisers.setup(opt, ps)

dct = Dict(:rng => rng, :model => model, :ps => ps, :st => st, :opt => opt, :st_opt => st_opt)

# Define loss function and training function
function loss_function(model, ps, data)
    global st   # st was already global, so why is global necessary?
    y_pred, _ = model(data, ps, st)  # Model should return size (model.out_dims, N). Update global st
    mse_loss = mean(abs2, y_pred)     # mutated data?
    return mse_loss, ()   # what is ()? 
end

function new_loss_function(model, data)
    global st, ps   # st was already global, so why is global necessary?
    mse_loss = mean(abs2, data)     # mutated data?
    return mse_loss, ()   # what is ()? 
end

function new_new_loss_function(model, data)
    global st, ps   # st was already global, so why is global necessary?
    y_pred, _ = model(data, ps, st)   # why does Julia think there are four arguments? 
    mse_loss = mean(abs2, y_pred)     # mutated data?
    return mse_loss, ()   # what is ()? 
end

# Mutation
Zygote.gradient((coef, data) -> loss_function(model, coef, data)[1], ps, randn(128,2))
# I do not understand the error on the next line. 
Zygote.gradient((coef) -> new_new_loss_function(model, coef)[1], ps)
# Works
res = Zygote.gradient((data) -> new_loss_function(model, data)[1], randn(128,2))

# Finally the training loop.
"""
function main(; model, ps, st, st_opt, data=nothing, epochs=10, kwargs...) #tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data::Tuple, epochs::Int)
    data = rand(128, 2)
    for epoch in 1:epochs
        # Mutation ERROR. So this must happen in the loss function!
        println(size(ps.coeffs))
        result = Zygote.gradient((coef, data) -> loss_function(model, coef, data)[1], ps, data)
        println("result**= ", length(result)) 
    end
end
"""

#main(; dct..., data=(x, y), epochs=5, dct[:ps].coeffs)
