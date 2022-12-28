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

`ps` might not be used
"""
function (l::Polylayer)(x::T, ps, st) where T <: AbstractArray{Float64} 
    if (l.out_dims == 1)
        poly = generate_polynomial_1D(l.degree)
        return poly(ps.coeffs, x), st
    elseif (l.out_dims == 2)
        x1 = x[:,1]
        x2 = x[:,2]

	    # Should ideally be generated only once and not every time I call the model
        # Coef generated within generated_polynomial. Should not be. 

        poly1 = generate_polynomial_2D(l.degree);
        poly2 = generate_polynomial_2D(l.degree);

        # Use views to reduce copying? Not worth is for small arrays
        coef1 = view(ps.coeffs, :, 1)
        coef2 = view(ps.coeffs, :, 2)
        p1 = poly1(coef1, x1, x2)
        p2 = poly2(coef2, x1, x2)
        return hcat(p1, p2), st
    else
        println("Poly_layer: l.out_dims > 2 not implemented")
    end
end

# ! The next four functions have the Lux prefix to make sure dynamic dispaching works properly
# overload initialparameters in the Lux environment
# coeffs[out_dims, degree+1]

# Probably unique in my project. 
# What happens if I do not prefix by Lux and I use the polylayer inside some other module. It is not clear this will work.
function nb_coefs_per_poly(d::Polylayer) 
    if d.out_dims == 1
        return d.degree+1
    elseif d.out_dims == 2
        return  Int((d.degree+1) * (d.degree+2) / 2.)  # (very specific). Works because there is no loss of precision
    else
        println("out_dims greater 2 not implemented.")
    end
end

function Lux.initialparameters(rng::AbstractRNG, d::Polylayer)
     nb = nb_coefs_per_poly(d)
     # why the final comma?
     if (d.out_dims == 1)
        return (coeffs=d.init_weight(rng, nb),)
     else 
        return (coeffs=d.init_weight(rng, nb, d.out_dims),)
     end
end

# overload initialstates in the Lux environment
Lux.initialstates(::AbstractRNG, ::Polylayer) = NamedTuple()

Lux.statelength(d::Polylayer) = 0

function Lux.parameterlength(d::Polylayer)
    return nb_coefs_per_poly(d) * d.out_dims
end
# ================== END DEFINITION of Polynomial Layer =======================================
# Simple standalone code to test the polynomial layer. Must be part of this file. 
function test_polynomial_layer(dim)
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    if (dim == 1)
        # 1D layer
        out_dims = 1
        degree = 3
        model = Polylayer(; out_dims=out_dims, degree=degree, init_weight=Lux.rand32)

        ps, st = Lux.setup(rng, model)
        ps = ComponentArray(ps)
        Lux.initialparameters(rng, model)

        Lux.ComponentArray(ps)
        x = rand(128)  # data

        # Evaluate the layer
        y, _ = model(x, ps, st)  # returns a single scalar
        return y

    elseif (dim == 2)
        # 2D layer
        out_dims = 2
        degree = 3
        model = Polylayer(; out_dims=out_dims, degree=degree, init_weight=Lux.rand32)

        ps, st = Lux.setup(rng, model)
        ps = ComponentArray(ps)
        Lux.initialparameters(rng, model)

        Lux.ComponentArray(ps)
        x = rand(128, 2)  # data

        # Evaluate the layer
        println("ps: ", ps.coeffs)
        y, _ = model(x, ps, st)
        return y
    end
end

y1     = test_polynomial_layer(1)  # ok
y1  = test_polynomial_layer(2)