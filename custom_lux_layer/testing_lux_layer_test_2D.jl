# 2022-12-25
# Get the poly layer working on a simple example, and then perhaps integeate this into 
# standalone_lux_layer.jl
#
# 2022-12-25
# Test that polynomial layer is working properly. 
# Create 2D polynomial and test the 2D polynomial layer
# The network weights are precisely the coefficients required. 
# The weights should be initialized to zero

# For more complex functions, say rational approximations, I should be able to create
# specialized layers on a case by case basis. This is not as easy as using Modeling Toolkit, 
# unless I use macros to create the approximate specilized layer, which could be done. 
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
function generate_data_2D(N)
    # Polynomial: P(x,y) = x^3 + 0.5 x y^2 - 2. y x^2 - (1.5 x) + .3
    # I think my problem is the poor choice of x and y. So let us take random numbers beteween -2 and 2 for both.
    # x and y ∈ [-2.,2.]
    x = -2.f0 .+ rand(N) .* 4. |> collect
    y = -2.f0 .+ rand(N) .* 4. |> collect
    # x = range(-2.f0, 2.f0, N) |> collect 
    # y = range(-2.f0, 2.f0, N) |> collect 
    z = zeros(2, N)
    # Coefficients: 1 x y x^2 xy y^2 x^3 (x^2 y) (x y^2) y^3
    #p = (a00=0.3, a10=-1.5, a01=0., a20=0., a11=0., a02=0.5,
         #a30=1., a21=-2., a12=0., a03=0.)
    #p = ComponentArray(p)
    @. z[1,:] = .3 - 1.5 * x - 2. * x^2 * y + 0.5 * x * y^2 + x^3
    @. z[2,:] =  x^2 - 2. * y^2  + 1.5

#    @. z[1,:] = .3  + 2. * x - 3. * y
#    @. z[2,:] =  -2. - x + 5. * y
    println("generate_data_2D, z= ",  size(z))
    # x,y: (1,N), z: (2,N)
    return x, y, z
end

N = 256
x_data, y_data, z_data = generate_data_2D(N)

model = Polylayer(; out_dims=2, degree=3, init_weight=Lux.zeros32)
ps, st = Lux.setup(rng, model)
#ps = Lux.initialparameters(rng, model)
ps = ComponentArray(ps)
opt = Lux.Adam(0.1f0)
# My cost function is not decreasing. Something is clearly wrong. 
st_opt = Optimisers.setup(opt, ps)
dct = Dict(:rng => rng, :model => model, :ps => ps, :st => st, 
           :opt => opt, :st_opt => st_opt, :N => N, 
           :x_data => x_data, :y_data => y_data, :z_data => z_data)

xy = hcat(x_data, y_data) |> transpose  # (2, N)

plot(z_data[1,:])
plot!(z_data[2,:])
# The model should return a vector of size 2 x 128. 
# The second value returned is the state, which is empty 

# model prediction is zero since all coefficients are initialized to zero
zz = model(xy, ps, st)[1]

function loss_function(model, ps, st, x_data, y_data, z_data, epoch)
    # x_data and y_data are 1D Vectors
    # println("loss,size x_data: ", size(x_data))  # 1D
    # println("loss,size z_data: ", size(z_data))  # 2D
    xy = hcat(x_data, y_data)   |> transpose# 
    z_pred, _ = model(xy, ps, st)  # Model should return size (model.out_dims, N). Update global st
    mse_loss  = mean(abs2, z_pred - z_data)     # mutated data?
    # induce sparsity
    lambda = .001
    mse_loss = mse_loss + lambda * norm(ps, 1)
    if (epoch % 10 == 0)
        # println("mse_loss[$(epoch)]: $(mse_loss), ps: $(ps)")
        println("mse_loss[$(epoch)]: $(mse_loss)")
    end
    return mse_loss, ()   # what is ()? 
end

#----------------------------------
function main(; model, ps, st, st_opt, x_data, y_data, z_data, epochs=200, N=128, kwargs...)
    # needed because st_opt on left-hand side further down, and a 
    # variable can only have a single declaration in a function
    # Next: need a call function. Perhaps use Optimization.jl package? 
    gs = 0
    for epoch in 1:epochs
        gs = Zygote.gradient(
                (coef, data) -> loss_function(model, coef, st, x_data, y_data, data, epoch)[1], 
                ps, z_data
            )[1]
        st_opt, ps = Optimisers.update(st_opt, ps, gs)  
    end
    return ps, gs
end

new_ps, gs = main(; dct..., epochs=500)

println("ps is (10,2). Should it be (2,10)?")
println("Am I at a local minimum or close to the global minimum?")

new_ps = collect(new_ps)
new_ps = reshape(new_ps, :, 2)
println("new_ps: $(new_ps[:,1])")
println("new_ps: $(new_ps[:,2])")

# The coefficients are wrong, although they appear to have the correct magnitudes, but associated
# wiht the wrong basis functions. 

"""
@. z[1,:] = .3 - 1.5 * x - 2. * x^2 * y + 0.5 * x * y^2 + x^3 ==> (.3, -1.5, 0., 0., 0,. 0., 1., -2., 0.5, 0.)
@. z[2,:] =  x^2 - 2. * y^2  + 1.5   ==> (1.5, 0., 0., 1., 0., -2., 0., 0., 0., 0.)
"""

"""
@. z[1,:] = .3  + 2. * x - 3. * y
@. z[2,:] =  -2. - x + 5. * y
"""