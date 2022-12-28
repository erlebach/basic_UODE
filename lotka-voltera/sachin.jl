using DifferentialEquations

module Support
D  = zeros(3, 3)
σ  = zeros(3, 3)

function eij(i, j)
    f =  zeros(3, 3)
    f[i,j] =  1.
    return f
end

E = Matrix{Matrix{Float64}}(undef, 3, 3)
for i in 1:3
    for j in 1:3
        E[i,j] = eij(i,j)
    end
end

# t can be a vector of times 
function gamma(t, γ₀, ω)
    #return γ₀ .* sin.(ω .* t)
    return γ₀ * sin(ω * t)
end

γ = gamma(3., 0.2, 3.)
γ̇ = gamma_dot(3., 0.2, 3.)

end

#---------------------------------------------------------
module Sachin
using DifferentialEquations
using Plots

function gamma_dot(t, γ₀, ω)
    return γ₀ * ω * cos(ω * t)
    #return γ₀ .* ω .* cos.(ω .* t)
end

mutable struct Model
    λ::Float64
    G::Float64
    γ₀::Float64
    ω::Float64
    ϵ::Float64
    α::Float64
end
model_UCM =  Model(.1, 1., 20., 100., 0.1, 0.1)
model_PTT =  Model(.1, 1., 20., 100., 0.1, 0.1)
model_G   =  Model(.1, 1., 20., 100., 0.1, 0.1)
f_UCM(m::Model) = [m.λ, m.G, m.γ₀, m.ω]  # function
f_PTT(m::Model) = [m.λ, m.G, m.γ₀, m.ω, m.ϵ]
f_G(m::Model) = [m.λ, m.G, m.γ₀, m.ω, m.α]
p_UCM = f_UCM(model_UCM)   # list
p_PTT = f_PTT(model_PTT)
p_G   = f_G(model_G)

function ODE_UCM!(du, u, p, t, gamma_dot)
    λ, G, γ₀, ω = p
    γ̇ = gamma_dot(t, γ₀, ω)
    σ11, σ22, σ33, σ12 = u
    du[1] = -(1. / λ) * σ11 + 2. * γ̇ * σ12
    du[2] = -(1. / λ) * σ22
    du[3] = -(1. / λ) * σ33
    du[4] = -(1. / λ) * σ12 + γ̇ * σ22 + G * γ̇
    return du
end # module

function ODE_PTT!(du, u, p, t, gamma_dot)
    λ, G, γ₀, ω, ϵ = p
    γ̇ = gamma_dot(t, γ₀, ω)
    σ11, σ22, σ33, σ12 = u
    trace = σ11 + σ22 + σ33
    fPTTλ = exp(ϵ * trace / G) / λ
    du[1] = -fPTTλ * σ11 + 2. * γ̇ * σ12
    du[2] = -fPTTλ * σ22
    du[3] = -fPTTλ * σ33
    du[4] = -fPTTλ * σ12 + γ̇ * σ22 + G * γ̇
    return du
end # module

# When is (ϵ*trace/G) == O(1)? 
# Answer: when $trace = G / ϵ)
# Trace approx 0.1, G = 1 ==> ϵ = G / trace = 10

function ODE_G!(du, u, p, t, gamma_dot)
    λ, G, γ₀, ω, α = p
    γ̇ = gamma_dot(t, γ₀, ω)
    σ11, σ22, σ33, σ12 = u
    coef = α / (λ * G) 
    du[1] = -σ11 / λ - coef * (σ11^2 + σ12^2) + 2. * γ̇ * σ12 
    du[2] = -σ22 / λ - coef * (σ22^2 + σ12^2)
    du[3] = -σ33 / λ - coef * σ33^2
    du[4] = -σ12 / λ - coef * ((σ11 + σ22) * σ12) + γ̇ * σ22 + G * γ̇
    return du
end # module

# Closure
Ncycles = 20
nb_per_cycle = 20 
T = Ncycles * (2. * π / model_UCM.ω)
tspan = (0., T)
u0 = [0., 0., 0., 0.] # (page 3 of Sachin project description [SPD22])
Δt = T / (Ncycles * nb_per_cycle)

function calculate_and_plot()
    dct = Dict(
        :λ => model_G.λ,
        :G => model_G.G,
        :γ₀ => model_G.γ₀,
        :ω => model_G.ω,
        :ϵ => model_G.ϵ,
        :α => model_G.α,
    )

    ODE_UCM_!(u, du, p, t) = ODE_UCM!(u, du, p, t, gamma_dot)
    ODE_PTT_!(u, du, p, t) = ODE_PTT!(u, du, p, t, gamma_dot)
    ODE_G_!(u, du, p, t) = ODE_G!(u, du, p, t, gamma_dot)

    function solve_ODEs(odes, u0, tspan, p, Δt)
        prob = ODEProblem(odes, u0, tspan, p)
        solution = solve(prob, Vern7(), abstol=1.e-7, reltol=1.e-7, saveat=Δt)
        dct = Dict()
        dct[:t] = solution.t
        dct[:X] = Array(solution)
        grid = (:xy, :olivedrab, :dot, 1, 0.9)
        dct[:plot] = plot(dct[:t], dct[:X]', grid=grid)
        return dct
    end

    dct_UCM = solve_ODEs(ODE_UCM_!, u0, tspan, p_UCM, Δt)
    dct_PTT = solve_ODEs(ODE_PTT_!, u0, tspan, p_PTT, Δt)
    dct_G   = solve_ODEs(ODE_G_!, u0, tspan, p_G, Δt)
    return dct_UCM, dct_PTT, dct_G
end

dct_UCM, dct_PTT, dct_G = calculate_and_plot()
plot(dct_UCM[:plot], dct_PTT[:plot], dct_G[:plot], layout=(3,1))

# printing the dictionary keys
# collect(keys(dct_PTT)
dct_PTT |> keys |> collect

coef_G = dct[:α] / (dct[:λ] * dct[:G]) 
X = dct_PTT[:X]
trace = X[1,:] + X[2,:] + X[3,:]
coef_PTT = exp(dct[:ϵ] * maximum(trace) / dct[:G]) / dct[:λ]
sol_ampl = (minimum(X), maximum(X))
dct
end
#---------------------------------------------------------
