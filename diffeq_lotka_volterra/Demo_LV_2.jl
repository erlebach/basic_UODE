

using DifferentialEquations
using Plots
using Flux
using DiffEqFlux

# Source: https://danpereda.github.io/post/scientificmachinelearning/

function lotka_volterra!(du, u, p, t)
    🐇, 🐺 = u
    α, β, γ, δ = p 
    du[1] = d🐇 = α*🐇 - β*🐇*🐺
    du[2] = d🐺 = γ*🐇*🐺 - δ*🐺
end

u₀ = [1.0, 1.0]
tspan = [0., 10.]
p = [1.5, 1., 3., 1.]
prob = ODEProblem(lotka_volterra!, u₀, tspan, p)
sol = solve(prob)

using Sundials
sol = solve(prob, CVODE_BDF(), save_everystep=false, saveat=0.1, abstol=1e-8, reltol=1e-8)

plot(sol)

remake(prob, p =[1.2,0.8,2.5,0.8])

#----------------------------------------------
# Universal ODE
function loss(p)
    tmp_prob = remake(prob, p=p)
    tmp_sol = solve(tmp_prob, saveat=0.1)
    sum(abs2, Array(tmp_sol) - dataset), tmp_sol
end

using Optim
pinit = [1.2, 0.8, 2.5, 0.8]
p
