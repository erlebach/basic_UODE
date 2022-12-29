"""
Arguments: x, y: data as 1D Vectors of size N, preallocated
"""
function generate_polynomial_2D(degree)
    # for 1D polynomials
    #IJ = [(i,j) for i in Ix for j in Iy if i + j < (degree+1)]
    #I = [i[1] for i in IJ]
    #J = [i[2] for i in IJ]

    if degree == 1
        I = (0,1,0)
        J = (0,0,1)
    elseif degree == 2
        #IJ = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
        I = (0,1,0,2,1,0)
        J = (0,0,1,0,1,2)
    elseif degree == 3
        #IJ = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)]
        I = (0, 1, 0, 2, 1, 0, 3, 2, 1, 0)
        J = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3)
    elseif degree == 4
        #IJ = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3), (4,0), (3,1), (2,2), (1,3), (0,4)]
        I = (0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0)
        J = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4)
    elseif degree == 5
        I = (0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0)
        J = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5)
    else
        println("Degree greater than 5 not implemented (2D degree: $(degree))")
    end

    # Construct a polynomial in x and y, according to the indices listed in I
    # Returns a function (coef, x, y) -> Array{T, N, out_dims) wehere T
    # Perhaps I could use the modeling toolkit to handle this?
    # Poly(coef, x, y) returns a Vector{T} where T is eltype(x)
     Poly(coef, x, y) = @tullio poly[i] := coef[j] * x[i]^I[j] * y[i]^J[j] grad=Dual nograd=x nograd=y nograd=I nograd=J
     
     return Poly
end

# Need better and more general approach
function generate_polynomial_1D(degree)
    # for 1D polynomials
    #IJ = [(i,j) for i in Ix for j in Iy if i + j < (degree+1)]
    #I = [i[1] for i in IJ]
    #J = [i[2] for i in IJ]

    if degree == 1
        I = (0,1)
    elseif degree == 2
        I = (0,1,2)
    elseif degree == 3
        I = (0, 1, 2, 3)
    elseif degree == 4
        I = (0, 1, 2, 3, 4)
    elseif degree == 5
        I = (0, 1, 2, 3, 4, 5)
    else
        println("Degree greater than 5 not implemented (1D degree: $(degree))")
    end

    # Construct a polynomial in x and y, according to the indices listed in I
    # Returns a function (coef, x, y) -> Array{T, N, out_dims)
     Poly(coef, x) = @tullio poly[i] := coef[j] * x[i]^I[j] grad=Dual nograd=x nograd=I 
     return Poly
end

function test_polynomial_generation_1d()
	N = 10000
	degree =3 
    x = rand(N)
    nb_terms = degree+1   # changes  dpending on out_dims
    coef = rand(nb_terms)
	poly = generate_polynomial_1D(degree);
	loss = (coef, x) -> sum(poly(coef, x).^2)
	result = Zygote.gradient(loss, coef, x)
    return result
end

function test_polynomial_generation_2d()
	N = 10000
	degree =3 
    x = rand(N)
    y = rand(N)
    nb_terms = Int(0.5*(degree+1)*(degree+2))   # changes  dpending on out_dims
    coef = rand(nb_terms)
	poly = generate_polynomial_2D(degree)
	loss = (coef, x, y) -> sum(poly(coef, x, y).^2)
	result = Zygote.gradient(loss, coef, x, y)
    return result
end


# Tests
result = test_polynomial_generation_1d();
result = test_polynomial_generation_2d();