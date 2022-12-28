macro imit(name::Symbol, value)
    return esc(quote
        if $name > $value
            $name = $value
        end
    end)
end

a = 5
@imit(a, 3)

macro seq(n::Symbol) 
    return esc(quote
        f = []
        I = Vector{Int}()
        if $n < 0
            print("The input must be equal to or bigger than zero.") 
        else  
            for i in 0:$n
                push!(I, i)
            end
        end
    I
    end)
end

d = 3
v = @seq d

macro IJ(n::Symbol) # n is degree
    return esc(quote
        f = []
        I = Vector{Int}()
        J = Vector{Int}()
        if $n < 0
            print("The input must be equal to or bigger than zero.") 
        else  
            for i in 0:$n
            for j in 0:$n
                if i+j <= $n
                    push!(I, i)
                    push!(J, j)
                end
            end
            end
        end
        (I, J)
    end)
end

function test()
    n = 5
    I, J = @IJ n
    println("I: ", I)
    println("J: ", J)
end

test()

d = 3
@IJ d