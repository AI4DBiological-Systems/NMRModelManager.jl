
"""
```
struct CostFuncBuffer{T, ST <: SubArray}
    X::BatchEvalBuffer{T}
    gs_re::Vector{Vector{T}}
    gs_im::Vector{Vector{T}}
    shift_multi_inds::Vector{NTuple{4,Int}}
    phase_multi_inds::Vector{NTuple{4,Int}}
    T2_multi_inds::Vector{NTuple{4,Int}}

    # buffers for computing the residual.
    b::Vector{Complex{T}}
    b_re::ST
    b_im::ST
end
```
"""
struct CostFuncBuffer{T, ST <: SubArray}
    X::BatchEvalBuffer{T}
    gs_re::Vector{Vector{T}}
    gs_im::Vector{Vector{T}}
    shift_multi_inds::Vector{NTuple{4,Int}}
    phase_multi_inds::Vector{NTuple{4,Int}}
    T2_multi_inds::Vector{NTuple{4,Int}}

    # buffers for computing the residual.
    b::Vector{Complex{T}}
    b_re::ST
    b_im::ST
end

function CostFuncBuffer(X::BatchEvalBuffer{T}, gr, gi, s, p, t)::CostFuncBuffer where T <: AbstractFloat
    N_samples = length(X.U_rad)

    b = Vector{Complex{T}}(undef, N_samples)
    r = reinterpret(T, b) # real(b[1]) imag(b[1]) real(b[2]) imag(b[2]) etc.
    b_re = view(r, 1:2:length(r)) # this is real.(b) with little allocation..
    b_im = view(r, 2:2:length(r)) # this is imag.(b) with little allocation.

    return CostFuncBuffer(X, gr, gi, s, p, t, b, b_re, b_im)
end

"""
```
CostFuncBuffer(MSS::CLMixtureSpinSys, U_rad)
```

Convenience constructor for type `CostFuncBuffer`.

- `U_rad` is the radial frequency (in radians) positions used for the cost function.
"""
function CostFuncBuffer(MSS::CLMixtureSpinSys, U_rad)

    mapping = SIG.ParamsMapping(MSS)

    return CostFuncBuffer(
        costfuncsetup(mapping, MSS, U_rad)...
    )
end

"""
```
function costfuncsetup(
    mapping::SIG.ParamsMapping,
    MSS::CLMixtureSpinSys,
    U_rad::AbstractArray{T},
)::Tuple{BatchEvalBuffer{T},Vector{Vector{T}},Vector{Vector{T}},Vector{NTuple{4,Int}},Vector{NTuple{4,Int}},Vector{NTuple{4,Int}}} where T
```

`U_rad` is in radians.
"""
function costfuncsetup(
    mapping::SIG.ParamsMapping,
    MSS::CLMixtureSpinSys{T, SST},
    U_rad::AbstractArray{T},
    )::Tuple{BatchEvalBuffer{T},Vector{Vector{T}},Vector{Vector{T}},Vector{NTuple{4,Int}},Vector{NTuple{4,Int}},Vector{NTuple{4,Int}}} where {T, SST}
    
    shift_multi_inds = createmappingLUT(mapping.shift)
    phase_multi_inds = createmappingLUT(mapping.phase)
    T2_multi_inds = createmappingLUT(mapping.T2)

    # buffers.
    X = BatchEvalBuffer(U_rad, MSS)

    gs_re = createbatchgradbuffer(MSS, length(U_rad))
    gs_im = createbatchgradbuffer(MSS, length(U_rad))

    return X, gs_re, gs_im, shift_multi_inds, phase_multi_inds, T2_multi_inds
end

"""
```
function evalcost!(
    costfunc_grad::Vector{T},
    model_params,
    p::Vector{T},
    C::CostFuncBuffer,
    y::Vector{Complex{T}},
)::T where T <: AbstractFloat
```

Evaluates the objective function at `p`. Also evaluates the graident, and stores in `C.X`.
The compound concentration `model_params.w` is treated as a constant here.

`y` is the data.
`p` is the model parameter input, the variable value to be evaluated at.
`model_params` is the data structure for the surrogate model.

"""
function evalcost!(
    costfunc_grad::Vector{T},
    model_params::SIG.MixtureModelParameters{T,MT},
    p::Vector{T},
    C::CostFuncBuffer,
    y::Vector{Complex{T}},
    )::T where {T <: AbstractFloat, MT <: SIG.CLMixtureSpinSys}

    @assert verifylen(info, p, model_params)
    
    X = C.X
    MSS = model_params.MSS

    importmodel!(model_params, p)

    updategradientbuffer!(X, MSS)
    constructdesignmatrix!(X, MSS)

    return evalcost!(
        costfunc_grad,
        model_params,
        C,
        y,
    )
end

function evalcost!(
    g::Vector{T},
    model_params::SIG.MixtureModelParameters{T,MT},
    C::CostFuncBuffer,
    y::Vector{Complex{T}},
    )::T where {T <: AbstractFloat, MT <: SIG.CLMixtureSpinSys}

    return evalcost!(g, SIG.AllVars(), model_params, C, y)
end

"""
```
evalcost!(
    grad_eval::Vector{T},
    info::SIG.VariableSetTrait,
    model_params::SIG.MixtureModelParameters{T,MT},
    C::CostFuncBuffer,
    y::Vector{Complex{T}},
)::T where {T <: AbstractFloat, MT <: SIG.CLMixtureSpinSys}
```

Evaluates the objective function at the flatten parameter storage in `model_params`. Also evaluates the graident, and stores in `C.X`.
The compound concentration `model_params.w` is treated as a constant here.

`y` is the data.
`model_params` is the data structure for the surrogate model.

"""
function evalcost!(
    grad_eval::Vector{T},
    info::SIG.VariableSetTrait,
    model_params::SIG.MixtureModelParameters{T,MT},
    C::CostFuncBuffer,
    y::Vector{Complex{T}},
    )::T where {T <: AbstractFloat, MT <: SIG.CLMixtureSpinSys}

    X, gs_re, gs_im = C.X, C.gs_re, C.gs_im
    shift_multi_inds, phase_multi_inds, T2_multi_inds = C.shift_multi_inds, C.phase_multi_inds, C.T2_multi_inds

    MSS = model_params.MSS
    w = model_params.w

    evalbatchgradient!(
        gs_re,
        gs_im,
        shift_multi_inds,
        phase_multi_inds,
        T2_multi_inds,
        MSS,
        w,
        X,
    )

    # residuals.
    #b = X.A *w - y
    #b_re = real.(b)
    #b_im = imag.(b)
    b = C.b
    mul!(b, X.A, w)
    axpy!(-one(T), y, b)

    computegrad!(grad_eval, info, C.b_re, gs_re, C.b_im, gs_im)

    #return norm(b)^2
    return dot(b,b)
end

function computegrad!(g::Vector{T}, ::SIG.AllVars, b_re, gs_re, b_im, gs_im)::Nothing where T
    
    for l in eachindex(g)
        g[l] = (dot(b_re, gs_re[l]) + dot(b_im, gs_im[l]))*2
    end

    return nothing
end

function computegrad!(g::Vector{T}, info::SIG.SubsetVars, b_re, gs_re, b_im, gs_im)::Nothing where T
    
    K = info.indices

    l = 0
    for i in K.shift
        l += 1
        g[l] = (dot(b_re, gs_re[i]) + dot(b_im, gs_im[i]))*2
    end

    for i in K.phase
        l += 1
        g[l] = (dot(b_re, gs_re[i]) + dot(b_im, gs_im[i]))*2
    end

    for i in K.T2
        l += 1
        g[l] = (dot(b_re, gs_re[i]) + dot(b_im, gs_im[i]))*2
    end

    return nothing
end

"""
```
function evalcost(
    model_params,
    p::Vector{T},
    y::Vector{Complex{T}},
    U_rad,
) where T <: AbstractFloat
```

Evaluates the objective function at the flatten parameter storage in `model_params`. Also evaluates the graident, and stores in `C.X`.
The compound concentration `model_params.w` is treated as a constant here.

`y` is the data.
`model_params` is the data structure for the surrogate model.
`p` is the model parameter input, the variable value to be evaluated at.
`U_rad` is the radial frequency locations (in radians) used in this objective function.

"""
function evalcost(
    model_params,
    p::Vector{T},
    y::Vector{Complex{T}},
    U_rad,
    ) where T <: AbstractFloat

    q = uu->evalmodel!(model_params, uu, p)

    cost = zero(T)
    for m in eachindex(U_rad)
        cost += abs2(q(U_rad[m]) - y[m])
    end

    return cost
end
