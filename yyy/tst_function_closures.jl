dct = Dict{Symbol, Any}(:a => 10, :b => 10.)
d = 50.

d = 65.

module modd 
function tst(dct)
    function tsta(a, b)
        dct[:c] = 45
    end
    c = 45
    dct[:tsta] = tsta
    print("gordon")
end

end

modd.tst(dct)