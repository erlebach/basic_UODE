# THIS LINE WAS ABANDONED. Use standalone_lux_layer_test.jl
# I will have to put the new poly_layer in a new file. 

using Revise
using Lux
using Random

#------------------------------------------------
module Layer
using Lux
using Random

"""
Apply a polynomial to an input Float
The coefficients are trainable. 
First attempt: polynomial of a specified `degree`
"""

struct Polylayer{F} <: Lux.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    degree::Int
    init_weight::F
end

function Base.show(io::IO, d::Polylayer)
    println(io, "Polylayer($(d.in_dims) => $(d.out_dims), degree: $(d.degree)")
    println("inside show")
end

function Polylayer(in_dims::Int, out_dims::Int, degree::Int, init_weight=Lux.rand32)
    dtypes = (typeof(init_weight))
    return Polylayer{dtypes}(in_dims, out_dims, degree, init_weight)
end

function initialparameters(rng::AbstractRNG, d::Polylayer)
    return (d.init_weight(rng, d.degree+1))
end

statelength(d::Polylayer) = 0

function parameterlength(d::Polylayer)
    return d.degree + 1
end

end    # end module
#------------------------------------------------------------------------

using 
l = Main.Layer
println(l.Polylayer)

rng = Random.default_rng()

model = l.Polylayer(5, 3, 3)

ps, st = Lux.setup(rng, model)

ps
