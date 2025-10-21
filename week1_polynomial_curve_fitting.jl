#=
The aim is to make an interactive plot to visualise polynomial curve fitting with different polynomial degrees,
training set sizes, noise level and regularisation parameters.
=#

using Plots
using Random

mutable struct Data  # should help in plotting (mostly)
    x::Vector{Float64}  # inp
    y::Vector{Float64}  # data output
    ŷ::Vector{Float64}  # predicted
    yₜ::Vector{Float64}  # true output (from function gen)
end

mutable struct Model
    epoch::Int  # epoch
    batch_size::Int  # batch_size (mini-batch size for SGD)
    n::Float64  # learning rate (using n instead of ita as both look the same)
    θ::Vector{Float64}  # weights
    λ::Float64  # L₂ regularisation term
    degree::Int  # polynomial degree
    x::Vector{Float64}  # inp
    y::Vector{Float64}  # output
end

function make_data(no_datapoints::Int, data_range::UnitRange, gen_func::Function = sin)::Data
    # Creates data
    x = zeros(no_datapoints)
    y = zeros(no_datapoints)

    st = first(data_range)
    ed = last(data_range)

    for iter in 1:no_datapoints
        x[iter] = st + rand() * (ed - st)  # provides a rand no with uniform distrib
        y[iter] = gen_func(x[iter]) + randn()  # standard uniform distribution
    end

    created_data = Data(x, y, Vector{Float64}(), Vector{Float64}())
    return created_data
end

function J(params::Model)
    # need a scalar here
    θ = params.θ
    x = params.x
    y = params.y
    degree = params.degree
    λ = params.λ

    Xₚ = x .^ transpose(0:degree)  # gets a matrix with all the powers till degree (no_inps x degree) - matrix and matrix interaction
    ŷ = Xₚ * θ  # gets a vector with ŷᵢ
    rigid_reg = (λ/2) * sum(θ[2:end] .^ 2)  # not penalise w_0 as it is just an intercept
    square_loss = sum((ŷ - y) .^ 2) / 2

    return square_loss + rigid_reg
end

function ∇J(params::Model)
    # can be figured out by matrix differentiation (partial based on each of the elements of weights)
    # it simplifies into ∇J = Xᵀ(Xθ - y) + λθ - have this instead of scalar, as it is for updating stuff, and not a scalar value
    θ = params.θ
    x = params.x
    y = params.y
    degree = params.degree
    λ = params.λ

    Xₚ = x .^ transpose(0:degree)  # similar to above
    
    square_loss_term = transpose(Xₚ)*(Xₚ*θ - y)
    rigid_reg_term = vcat(0.0, λ .* θ[2:end])  # do not penalise θ[1]

    return square_loss_term + rigid_reg_term
end

function gradient_decent(params::Model)
    n = params.n
    epoch = params.epoch
    batch_size = params.batch_size
    init_loss = J(params)

    for iter in 1:epoch
        loss = J(params)
        if loss < 1e-3
            break
        end
        println("Epoch: $iter, Loss: $loss")
        params.θ -= n * ∇J(params)
    end
end