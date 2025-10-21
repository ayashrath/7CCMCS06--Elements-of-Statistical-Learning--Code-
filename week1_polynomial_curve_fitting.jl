#=
The aim is to make an interactive plot to visualise polynomial curve fitting with different polynomial degrees,
training set sizes, noise level and regularisation parameters.
=#

using CairoMakie

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

function make_data(no_datapoints::Int, data_range::UnitRange, gen_func::Function = sin, mean::Float64 = 0.0, std::Float64 = 1.0)::Data
    # Creates data
    x = zeros(no_datapoints)
    y = zeros(no_datapoints)
    yₜ = zeros(no_datapoints)

    st = first(data_range)
    ed = last(data_range)
    
    x = collect(LinRange(st, ed, no_datapoints))

    # doing it in another iter as we want the x to be ordered
    for iter in 1:no_datapoints        
        y[iter] = gen_func(x[iter]) + mean + randn() * std  # standard uniform distribution
        yₜ[iter] = gen_func(x[iter])  # the real data (if noise was not added)
    end

    created_data = Data(x, y, Vector{Float64}(), yₜ)
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

function ∇J(params::Model, ind_array::Vector{Int})
    # can be figured out by matrix differentiation (partial based on each of the elements of weights)
    # it simplifies into ∇J = Xᵀ(Xθ - y) + λθ - have this instead of scalar, as it is for updating stuff, and not a scalar value
    θ = params.θ
    x = params.x[ind_array]
    y = params.y[ind_array]
    degree = params.degree
    λ = params.λ

    Xₚ = x .^ transpose(0:degree)  # similar to above
    
    square_loss_term = transpose(Xₚ)*(Xₚ*θ - y) / length(ind_array)  # mini_batch implemented
    rigid_reg_term = vcat(0.0, λ .* θ[2:end])  # do not penalise θ[1]

    return square_loss_term + rigid_reg_term
end

function gradient_decent(params::Model)
    n = params.n
    epoch = params.epoch
    batch_size = params.batch_size
    loss = J(params)
    len_x = length(params.x)

    θ_best = copy(params.θ)
    best_loss = J(params)
    prev_loss = J(params)

    for iter in 1:epoch
        loss = J(params)
        perm = randperm(len_x)  # creates a random permulation within the size limit
        
        if loss < 1e-3  || loss / prev_loss < 1e-3  # when it converges or is good enough
            println("Converging! Epoch: $iter, Loss: $loss, Learning Rate: $n")
            break
        elseif loss / best_loss  > 1.5
            println("Reducing Learning Rate! Epoch: $iter, Loss: $loss, Learning Rate: $n")
            n *= 0.5
            params.θ = copy(θ_best)
        end

        if iter % div(epoch, 10) == 0  || iter == 1
            println("Epoch: $iter, Loss: $loss, Learning Rate: $n")
            # println("Best loss: $best_loss")
        end
        
        if best_loss > loss
            best_loss = loss
            θ_best = copy(params.θ)
        end

        for iter₂ in 1:batch_size:len_x
            ind_array = perm[iter₂:min(iter₂ + batch_size - 1)]  # gets the part you want (min used to account for case when it might get out of bounds)
            params.θ -= n * ∇J(params, ind_array)
            if J(params) / best_loss  > 1.5
                println("Reducing Learning Rate! Epoch: $iter, Loss: $loss, Learning Rate: $n")
                n *= 0.5
                params.θ = copy(θ_best)
                break
            end
        end
    end
    
    println("Epoch: $epoch, Loss: $loss, Learning Rate: $n")
    println("Best loss: $best_loss")
    print("Weights: $(params.θ)")
end

function predict(params::Model)
    θ = params.θ
    x = params.x
    degree = params.degree

    Xₚ = x .^ transpose(0:degree)  # gets a matrix with all the powers till degree (no_inps x degree) - matrix and matrix interaction
    ŷ = Xₚ * θ

    return ŷ
end


function main()
    # the main stuff
    no_datapoints = 1000
    data_range = -10:10
    gen_function = x -> x^2
    mean = 0.0
    std = 0.0

    epoch = 50
    batch_size = no_datapoints
    n = 1
    λ = 0
    degree = 2
    θ_init = zeros(degree + 1)

    # Computing
    data = make_data(no_datapoints, data_range, gen_function, mean, std)
    model = Model(epoch, batch_size, n, θ_init, λ, degree, data.x, data.y)
    gradient_decent(model)
    data.ŷ = predict(model)
    current_loss = J(model)


    # Does the plots for now
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        title = "Polynomial Regression with SGD (Epoch: $epoch, Degree: $degree, Loss: $(round(current_loss)))"
    )
    scatter!(
        ax,
        data.x,
        data.y,
        color = :blue,
        label = "Data Points",
    )
    lines!(
        ax, 
        data.x, 
        data.yₜ,
        color = :tomato,
        linestyle = :dash,
        label = "True (no noise)",
    )
    lines!(
        ax,
        data.x,
        data.ŷ,
        color = :black,
        label = "Prediction"
    )
    axislegend(position = :rt)
    fig
end

main()