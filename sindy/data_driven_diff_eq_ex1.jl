using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using DataDrivenSparse
using DataDrivenSR
using LinearAlgebra
using Plots

# Create a test problem
function lorenz(u,p,t)
    x, y, z = u

    ẋ = 10.0*(y - x)
    ẏ = x*(28.0-z) - y
    ż = x*y - (8/3)*z
    return [ẋ, ẏ, ż]
end

u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
dt = 0.1
prob = ODEProblem(lorenz,u0,tspan)
sol = solve(prob, Vern7(), saveat = dt)
plot(sol)

## Start the automatic discovery
ddprob = DataDrivenProblem(sol)

@variables t x(t) y(t) z(t)
u = [x;y;z]
basis = Basis(polynomial_basis(u, 5), u, iv = t)
opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 3))
println(get_basis(ddsol))
ggg = get_basis(ddsol)
println(get_parameter_values(ggg))
println(ddsol.residuals)  # 3.71 (not converged)
# Residuals around 68, so the apparently non-converged. However, the basis coefficients look correct
# WHY IS THAT? 