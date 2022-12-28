using Zygote
using Tullio

function generate_polynomial(N, degree, x, y)
    println("Enter Polynomial::generate_polynomial, length x, y: ", size(x), size(y)) # x: (128, 2), y: (2, 128)

    Ix = 0:degree
    Iy = 0:degree

    # for 1D polynomials
    IJ = [(i,j) for i in Ix for j in Iy if i + j < (degree+1)]
    I = [i[1] for i in IJ]
    J = [i[2] for i in IJ]

    # Construct a polynomial in x and y, according to the indices listed in I
     Poly(coef) = @tullio poly[i] := coef[j] * x[i]^I[j] * y[i]^J[j] grad=Dual nograd=x nograd=y nograd=I nograd=J

    # 1-D polynomial
    #Poly(coef) = @tullio poly[i] := coef[j] * x[i]^(I[j][1])  grad=Dual nograd=x nograd=I verbose=2
    #Poly(coef) = @tullio poly[i] := coef[j] * x[i]^(I[j][1])  grad=Dual verbose=2
    
    #works, but not appropriate for a more general task
    #Poly(coef) = @tullio z[i] := coef[n+1, m+1] * x[i]^n * y[i]^m  grad=Dual nograd=x nograd=y

    return Poly 
end

N = 128
degree = 4 # single term
#degree = 1 # three terms
x = rand(N)
y = rand(N)
#poly, coef = generate_polynomial(N, degree, x, y);
poly = generate_polynomial(N, degree, x, y);
nb_terms = Int((degree+1)*(degree+2) / 2)
#nb_terms = degree + 1
coef = rand(nb_terms)
loss = x -> sum(poly(x).^2)
Zygote.gradient(loss, coef)

# The following works
I = [(1,2), (2,3)]
x = rand(5)
loss = x -> sum(x.^(I[1][2]) + x.^(I[2][1]))
Zygote.gradient(loss, x)