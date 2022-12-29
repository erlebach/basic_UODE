N = 128
x = rand(1, N)
coef = rand(1, N)
#coef_pred, st = model(x_data, ps, st)
# Evaluate the predicted polynomial (degree 3)
x1 = reshape(x, :)
x2 = reshape(x.^ 2, :)
x3 = reshape(x.^ 3, :)
pred_poly = coef[1] .+ coef[2] .* x1 .+ coef[3] .* x2 .+ coef[4] .* x3
loss = mean( (y_data .-pred_poly) .^ 2)
println("loss: ", loss)



N = 128
x = rand(1, N)
coef = rand(4, N)
#coef_pred, st = model(x_data, ps, st)
# Evaluate the predicted polynomial (degree 3)
x2 = x.^ 2
x3 = x.^ 3
pred_poly = coef[1,:] .+ coef[2,:] .* x1 .+ coef[3] .* x2 .+ coef[4] .* x3
loss = mean( (y_data .-pred_poly) .^ 2)
println("loss: ", loss)


