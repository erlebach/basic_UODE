2022-12-27
TODO: Verify that the polynomial layer is operating as expected. 
Input to the network in "testing_lux-layer_test_2D.jl" is the array x. 
The output of the network are the polynomial values. 
The network weights are the polymomial coefficients. For 1D polynomials, there are d+1 coefficients. In 2D, there are (d+1)*(d+2)/2 coefficients. `d` is the total degree, so that the sum of the degrees of the different variables cannot exceed d+1. 
------------------------------------------------------------------
2022-12-28
Done: demonstrated a 1D polynomial layer: one function, one polynomial. 
TODO: solve a 1D equations of the form
  dy/dt = -y + .1 y^2 - .3 y^4
using UODE in the form: 
  dy/dt = -y + NN(y)
where NN is a neural network with one input and one output.
Starting point: testing_lux_layer_test_1D.jl
File name: 1D_equation_with_poly_layer_1D.jl
------------------------------------------------------------------
2022-12-28
New code: 