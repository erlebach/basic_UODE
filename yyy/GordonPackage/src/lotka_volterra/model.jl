
function construct_model(layer_size, act)
    dense = Lux.Dense
    chain = Lux.Chain

    model = chain(dense(2, layer_size, act),
              dense(layer_size, layer_size, act),
              dense(layer_size, layer_size, act),
              dense(layer_size, 2))
    return model
end
