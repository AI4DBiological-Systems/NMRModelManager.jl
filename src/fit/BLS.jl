
#### utilities
function reinterpretcomplexvector(y::Vector{Complex{T}})::Vector{T} where T <: AbstractFloat
    return reinterpret(T, y)
end

######### BLS


struct BLSParameters{T <: AbstractFloat}
    P_vec::Vector{T} # buffer.
    #A::Matrix{Complex{T}}
    optim_prob
    primal_initial::Vector{T}
    dual_initial::Vector{T}
    observations::Vector{T}
    lbs::Vector{T}
    ubs::Vector{T}
end

@kwdef struct BLSConfig{T}
    # eps_abs::T = convert(T, 1e-12)
    # eps_rel::T = convert(T, 1e-8)
    eps_abs::T = convert(T, 1e-7)
    eps_rel::T = convert(T, 1e-6)
    max_iter::Int = 4000
    verbose::Bool = false
    alpha::T = one(T)
end


###### package up BLS. The update BLS thing doesn't seem to be working well.
# just re-setup every time we solve.

# no bound checking on v.
# upper triangule of A, including the diagonal.
function updateuppertriangular!(v, A)

    N = size(A,1)
    resize!(v, div(N*(N-1),2)+N) 

    k = 1
    for j in axes(A,2)
        for i in axes(A,1)[begin:begin+j-1]
            v[k] = A[i,j]
            k += 1
        end
    end

    return nothing
end

function setupBLS(
    #mat_params,
    B::Matrix{T},
    y::Vector{T}, # reinterpretcomplexvector(y)
    lbs::Vector{T},
    ubs::Vector{T},
    config::BLSConfig{T};
    adaptive_rho = true,
    ) where T <: AbstractFloat

    eps_abs, eps_rel, max_iter, verbose, alpha = config.eps_abs, config.eps_rel, config.max_iter, config.verbose, config.alpha
    
    # set up model objects.
    N = size(B,2)
    @assert length(lbs) == length(ubs) == N
    
    @assert length(y) == size(B,1)

    # turn them into quadratic program objects. OSQP seems to only take Float64 arrays.
    P = sparse(convert(Matrix{Float64}, B'*B))
    q = convert(Vector{Float64}, -B'*y)
    G = sparse(LinearAlgebra.I, N, N)

    # buffer for P, for updating purpose.
    P_vec_buf = zeros(T, div(N*(N-1),2)+N) # upper triangule of P, including the diagonal.
    #updateuppertriangular!(P_vec_buf, P)

    # create the problem.
    prob = OSQP.Model()

    OSQP.setup!(
        prob;
        P = P, 
        q = q, 
        A = G, 
        l = convert(Vector{Float64}, lbs),
        u = convert(Vector{Float64}, ubs),
        alpha = alpha,
        eps_abs = eps_abs,
        eps_rel = eps_rel,
        max_iter = max_iter,
        verbose = verbose,
        adaptive_rho = adaptive_rho,
    )

    # assemble parameters and problem into a data structure.
    primal_initial = convert(Vector{T}, (lbs+ubs) ./2)
    dual_initial = zeros(T, N)
    BLS_params = BLSParameters(
        P_vec_buf,
        #A_buf,
        prob,
        primal_initial,
        dual_initial,
        y,
        lbs,
        ubs,
    )

    return BLS_params
end

# OSQP docs: https://osqp.org/docs/interfaces/julia.html#solve
function solveBLSold!(
    BLS_params::BLSParameters{T},
    #mat_params,
    B::Matrix{T},
    ) where T <: AbstractFloat

    prob, P_vec_buf, y = BLS_params.optim_prob, BLS_params.P_vec, BLS_params.observations
    primal_initial= BLS_params.primal_initial
    dual_initial = BLS_params.dual_initial

    @assert length(primal_initial) == length(dual_initial)

    # update problem.
    #B = constructdesignmatrix!(mat_params)
    P = B'*B
    updateuppertriangular!(P_vec_buf, P)
    q_new = -B'*y

    OSQP.update!(prob, Px = P_vec_buf, q = q_new)

    # solve.
    OSQP.warm_start!(prob; x = primal_initial, y = dual_initial)
    results = OSQP.solve!(prob)

    # prepare the results.
    primal_sol = convert(Vector{T}, results.x)
    dual_sol = convert(Vector{T}, results.y)

    status_flag = true
    if results.info.status != :Solved
        status_flag = false
    end

    obj_val = convert(T, results.info.obj_val)

    return primal_sol, dual_sol, status_flag, obj_val
end

function solveBLS!(
    BLS_params::BLSParameters{T},
    B::Matrix{T},
    ) where T <: AbstractFloat

    lbs, ubs = BLS_params.lbs, BLS_params.ubs
    y = BLS_params.observations
    w_LS, skip_solver = getBLSinitial(B, y, lbs, ubs)

    if !skip_solver
        for d in eachindex(w_LS)
            clamp(w_LS[d], lbs[d], ubs[d])
        end
        return solveBLSOSQP(B, y, lbs, ubs; primal_initial = w_LS)
    end

    return w_LS, true
end

# the second output is true if the solution w is within bounds: lbs, ubs.
function getBLSinitial(
    B::Matrix{T},
    y::Vector{T},
    lbs::Vector{T},
    ubs::Vector{T}
    )::Tuple{Vector{T},Bool} where T
    
    w = B\y

    for i in eachindex(w)
        if !(lbs[i] <= w[i] <= ubs[i])
            return w, false
        end
    end

    return w, true
end

function solveBLSOSQP(B::Matrix{T}, y::Vector{T}, lbs, ubs; primal_initial = lbs)::Tuple{Vector{T},Bool} where T
    
    P = B'*B
    q = -B'*y

    results = runOSQPreference(
        P,
        q,
        lbs,
        ubs;
        primal_initial = primal_initial,
    )

    # prepare the results.
    primal_sol = convert(Vector{T}, results.x)
    
    status_flag = true
    if results.info.status != :Solved
        status_flag = false
    end

    return primal_sol, status_flag
end

# creates OSQP object from scratch, and solves.
function runOSQPreference(
    P::Matrix{T},
    q::Vector{T},
    lbs::Vector{T},
    ubs::Vector{T};
    primal_initial::Vector{T} = lbs,
    verbose = false,
    ) where T

    N = length(lbs)
    @assert length(ubs) == N
    
    # Crate OSQP object
    prob2 = OSQP.Model()
    OSQP.setup!(
        prob2;
        P = sparse(convert(Matrix{Float64}, P)),
        q = convert(Vector{Float64}, q),
        A = sparse(LinearAlgebra.I, N, N),
        l = convert(Vector{Float64}, lbs),
        u = convert(Vector{Float64}, ubs),
        alpha = convert(Float64, 1.6),
        verbose = verbose,
    )

    # Solve problem
    OSQP.warm_start!(prob2; x = convert(Vector{Float64}, primal_initial))
    return OSQP.solve!(prob2)
end

########## for testing

# # test reinterpret on Matrix{Complex{T}}. it should be an interlaced matrix of double the number of rows.
# function interlacematrix(A::Matrix{Complex{T}})::Matrix{T} where T <: AbstractFloat

#     B = Matrix{T}(undef, size(A,1)*2, size(A,2))
#     for r in axes(A, 1)
#         for c in axes(A, 2)
#             B[2*(r-1)+1, c] = real(A[r,c])
#             B[2*(r-1)+2, c] = imag(A[r,c])
#         end
#     end

#     return B
# end



###### model-specific.


function setupwsolver(X, MSS, lbs::Vector{T}, ubs::Vector{T}, y::Vector{Complex{T}}, config::BLSConfig{T}) where T

    N = size(X.A, 2)

    @assert N == length(lbs) == length(ubs)

    BLS_params = setupBLS(
        constructdesignmatrix!(
            UseGradient(),
            X,
            MSS,
        ),
        reinterpretcomplexvector(y),
        lbs,
        ubs,
        config,
    )

    return BLS_params
end

function solvew!(
    MSS::SIG.CLMixtureSpinSys,
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients.
    X::BatchEvalBuffer{T},
    ) where T <: AbstractFloat

    primal_sol, status_flag = solveBLS!(
        BLS_params,
        constructdesignmatrix!(X, MSS),
    )
    w = collect(
        clamp(
            primal_sol[i],
            BLS_params.lbs[i],
            BLS_params.ubs[i]
        ) for i in eachindex(BLS_params.ubs)
    )
    BLS_params.primal_initial[:] = w

    return w
end

function solvew!(
    MSS::SIG.CLMixtureSpinSys,
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients.
    X::BatchEvalBuffer{T},
    active_systems::Vector{Tuple{Int,Int}},
    ) where T <: AbstractFloat

    primal_sol, status_flag = solveBLS!(
        BLS_params,
        constructdesignmatrix!(X, MSS, active_systems),
    )
    w = collect(
        clamp(
            primal_sol[i],
            BLS_params.lbs[i],
            BLS_params.ubs[i]
        ) for i in eachindex(BLS_params.ubs)
    )
    BLS_params.primal_initial[:] = w

    return w
end

function updatew!(
    model_params,
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients.
    X::BatchEvalBuffer{T},
    ::IgnoreGradient,
    p::Vector{T},
    info::SIG.VariableSetTrait,
    ) where T <: AbstractFloat

    MSS = model_params.MSS

    #SIG.importmodelreset!(model_params, p, SIG.getindices(info))
    SIG.importmodel!(model_params, p, SIG.getindices(info))
    updatebuffer!(X, MSS, SIG.getactivesystems(info))

    w = solvew!(MSS, BLS_params, X)
    model_params.w[:] = w

    return nothing
end

function updatew!(
    model_params,
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients.
    X::BatchEvalBuffer{T},
    ::UseGradient,
    p::Vector{T},
    info::SIG.VariableSetTrait,
    ) where T <: AbstractFloat

    MSS = model_params.MSS

    #SIG.importmodelreset!(model_params, p, SIG.getindices(info))
    SIG.importmodel!(model_params, p, SIG.getindices(info))
    updategradientbuffer!(X, MSS, SIG.getactivesystems(info))

    w = solvew!(MSS, BLS_params, X)
    model_params.w[:] = w

    return nothing
end

function verifylen(info::SIG.SubsetVars, p::Vector{T}, args...)::Bool where T <: AbstractFloat
    return length(p) == SIG.getNvars(info)
end

function verifylen(::SIG.AllVars, p::Vector{T}, model_params::SIG.MixtureModelParameters)::Bool  where T <: AbstractFloat
    return length(p) == SIG.getNvars(model_params)
end


# creates own version of w.
# every interation, solve for BLS.
function evalenvelopecost!(
    model_params,
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients.
    C::CostFuncBuffer,
    p::Vector{T},
    y::Vector{Complex{T}},
    info::SIG.VariableSetTrait,
    )::T where T <: AbstractFloat

    @assert verifylen(info, p, model_params)

    X = C.X
    updatew!(model_params, BLS_params, X, IgnoreGradient(), p, info)
    #updatew!(model_params, BLS_params, X, UseGradient(), p)
    w = model_params.w

    #return norm(X.A *w - y)^2
    b = C.b
    mul!(b, X.A, w)
    axpy!(-one(T), y, b)

    return dot(b,b)
end

function evalenvelopecost!(
    model_params,
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients.
    C::CostFuncBuffer,
    p::Vector{T},
    y::Vector{Complex{T}},
    )::T where T <: AbstractFloat

    return evalenvelopecost!(model_params, BLS_params, C, p, y, SIG.AllVars())
end

function evalenvelopegradient!(
    grad_p::Vector{T}, # mutates.
    model_params, # mutates.
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients. mutates.
    C::CostFuncBuffer, # mutates.
    p::Vector{T},
    y::Vector{Complex{T}},
    info::SIG.VariableSetTrait,
    )::T where T <: AbstractFloat

    @assert verifylen(info, p, model_params)

    updatew!(model_params, BLS_params, C.X, UseGradient(), p, info)

    # cost.
    return evalcost!(
        grad_p,
        info,
        model_params,
        C,
        y,
    )
end

function evalenvelopegradient!(
    grad_p::Vector{T}, # mutates.
    model_params, # mutates.
    BLS_params::BLSParameters{T}, # BLS inputs. BLS solves for the basis coefficients. mutates
    C::CostFuncBuffer, # mutates.
    p::Vector{T},
    y::Vector{Complex{T}},
    ) where T <: AbstractFloat

    return evalenvelopegradient!(grad_p, model_params, BLS_params, C, p, y, SIG.AllVars())
end