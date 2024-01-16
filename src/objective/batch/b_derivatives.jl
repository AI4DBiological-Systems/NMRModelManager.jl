
#### utilities.

function createmappingLUT(mapping::SIG.MoleculeParamsMapping)

    N = 0
    for n in eachindex(mapping.st)
        for i in eachindex(mapping.st[n])
            for j = mapping.st[n][i]:mapping.fin[n][i]
                N += 1
            end
        end
    end

    multi_inds = Vector{Tuple{Int,Int,Int,Int}}(undef, N)
    lin_ind = 0
    for n in eachindex(mapping.st)
        for i in eachindex(mapping.st[n])

            j = 0 # index for Coherence parameters, i.e. loops over the dimension of all Δc_bar's of spin system (n,i).
            for l = mapping.st[n][i]:mapping.fin[n][i] # l is the content of `mapping`.
                lin_ind += 1
                j += 1

                multi_inds[lin_ind] = (n,i,l,j)
            end
        end
    end

   return multi_inds
end

function createbatchgradbuffer(
    MSS::CLMixtureSpinSys{T,ST,PT,T2T},
    M::Int
    ) where {T,ST,PT,T2T}

    N_vars = SIG.getNvars(MSS)
    gs = Vector{Vector{T}}(undef, N_vars)
    for i in eachindex(gs)
        gs[i] = ones(T, M) .* NaN
        #gs[i] = ones(T, M) .* Inf
    end

    return gs
end

#### derivatives.

# diff approach.
function evalbatchgradient!(
    gs_re::Vector{Vector{T}},
    gs_im::Vector{Vector{T}},
    shift_multi_inds,
    phase_multi_inds,
    T2_multi_inds,
    MSS::CLMixtureSpinSys{T, ST, PT, T2T},
    w::Vector{T},
    X::BatchEvalBuffer{T},
    ) where {T <: AbstractFloat, ST, PT, T2T}

    @assert length(gs_re) == length(gs_im)
    
    # mandatory: reset gradient accumulation buffers to zero.
    for i in eachindex(gs_re)
        fill!(gs_re[i], zero(T))
        fill!(gs_im[i], zero(T))
    end

    #@show all(isfinite.(gs_re[10])) # debug.
    #@assert all( all(isfinite.(gs_re[i])) for i in eachindex(gs_re))
    #@assert all( all(isfinite.(gs_im[i])) for i in eachindex(gs_im))

    # get gradient.
    evalshiftgradient!(
        gs_re,
        gs_im,
        shift_multi_inds,
        MSS,
        w,
        X,
    )
    
    evalphasegradient!(
        gs_re,
        gs_im,
        phase_multi_inds,
        MSS,
        w,
        X,
    )

    evalT2gradient!(
        gs_re,
        gs_im,
        T2_multi_inds,
        MSS,
        w,
        X,
    )
    # @show all(isfinite.(gs_re[10])) # debug.
    # @show all(isfinite.(gs_re[10])) # debug.
    # println() # debug.
    #@assert all( all(isfinite.(gs_re[i])) for i in eachindex(gs_re))
    #@assert all( all(isfinite.(gs_im[i])) for i in eachindex(gs_im))

    return nothing
end

####### for a given type of parameter, i.e. shift, phase, T2. Loop through the type by index l.

# identical to the other evalshiftgradient!() except calling evalSharedShiftgradient!() and not passing ηs[j][k] in inner loop.
# not merging evalshiftgradient!() of SharedShift with
# evalshiftgradient!() of CoherenceShift due to possible performance penalty for passing the unused ηs.
# revisit whether to merge these after model fitting code is implemented and testing, to see how bad the penalty for merging is.
function evalshiftgradient!(
    gs_re::Vector{Vector{T}},
    gs_im::Vector{Vector{T}},
    shift_multi_inds,
    MSS::CLMixtureSpinSys{T, SIG.SharedShift{T}, SIG.CoherencePhase{T}, T2T},
    w::Vector{T},
    X::BatchEvalBuffer{T},
    ) where {T <: AbstractFloat, T2T}

    Δc_bars, phases =  MSS.Δc_bars, MSS.phases

    # shift.
    for (n,i,l,j) in shift_multi_inds
        
        ηs = Δc_bars[n][i]
        g_re = gs_re[l]
        g_im = gs_im[l]

        ∂frs = X.grad_re_evals[n][i]
        ∂fis = X.grad_im_evals[n][i]

        for k in eachindex(ηs)

            c, s = SIG.fetchphase(phases, n, i, k)
            
            evalSharedShiftgradient!(
                g_re,
                g_im,
                ∂frs[k],
                ∂fis[k],
                c,
                s,
            )

        end # end m loop.

        mulw!(g_re, g_im, w[n])
    end # end shift loop.

    return nothing
end

function evalshiftgradient!(
    gs_re::Vector{Vector{T}},
    gs_im::Vector{Vector{T}},
    shift_multi_inds,
    MSS::CLMixtureSpinSys{T, SIG.CoherenceShift{T}, SIG.CoherencePhase{T}, T2T},
    w::Vector{T},
    X::BatchEvalBuffer{T},
    ) where {T <: AbstractFloat, T2T}

    Δc_bars, phases =  MSS.Δc_bars, MSS.phases

    # shift.
    for (n,i,l,j) in shift_multi_inds
        
        ηs = Δc_bars[n][i]
        g_re = gs_re[l]
        g_im = gs_im[l]

        ∂frs = X.grad_re_evals[n][i]
        ∂fis = X.grad_im_evals[n][i]

        for k in eachindex(ηs)

            c, s = SIG.fetchphase(phases, n, i, k)
            
            evalCoherenceShiftgradient!(
                g_re,
                g_im,
                ∂frs[k],
                ∂fis[k],
                c,
                s,
                ηs[k][j],
            )

        end # end m loop.

        mulw!(g_re, g_im, w[n])
    end # end shift loop.

    return nothing
end

function evalphasegradient!(
    gs_re::Vector{Vector{T}},
    gs_im::Vector{Vector{T}},
    phase_multi_inds,
    MSS::CLMixtureSpinSys{T, ST, SIG.CoherencePhase{T}, T2T},
    w::Vector{T},
    X::BatchEvalBuffer{T},
    ) where {T <: AbstractFloat, ST, T2T}

    Δc_bars, phases =  MSS.Δc_bars, MSS.phases

    # phase.
    for (n,i,l,j) in phase_multi_inds

        ηs = Δc_bars[n][i]
        g_re = gs_re[l]
        g_im = gs_im[l]

        frs = X.re_evals[n][i]
        fis = X.im_evals[n][i]

        for k in eachindex(ηs)
            
            c, s = SIG.fetchphase(phases, n, i, k)
            
            evalCoherencePhasegradient!(
                g_re,
                g_im,
                frs[k],
                fis[k],
                c,
                s,
                ηs[k][j],
            )

        end # end m loop.

        mulw!(g_re, g_im, w[n])
    end # end phase loop.

    return nothing
end

function evalT2gradient!(
    gs_re::Vector{Vector{T}},
    gs_im::Vector{Vector{T}},
    T2_multi_inds,
    MSS::CLMixtureSpinSys{T, ST, SIG.CoherencePhase{T}, SIG.SharedT2{T}},
    w::Vector{T},
    X::BatchEvalBuffer{T},
    ) where {T <: AbstractFloat, ST}

    phases, λ0 = MSS.phases, MSS.λ0

    # T2.
    for (n,i,l,_) in T2_multi_inds

        g_re = gs_re[l]
        g_im = gs_im[l]

        # # passes.
        # @assert all(isfinite.(g_re))
        # @assert all(isfinite.(g_im))

        ∂frs = X.grad_re_evals[n][i]
        ∂fis = X.grad_im_evals[n][i]

        for k in eachindex(∂frs)
            
            c, s = SIG.fetchphase(phases, n, i, k)
            
            evalSharedT2gradient!(
                g_re,
                g_im,
                ∂frs[k],
                ∂fis[k],
                c,
                s,
                λ0,
            )

        end # end m loop.

        # @assert all(isfinite.(g_re)) # fails.
        # @assert all(isfinite.(g_im)) # fails.

        mulw!(g_re, g_im, w[n])
    end # end T2 loop.

    return nothing
end

########## for a given parameter.

function mulw!(g_re::Vector{T}, g_im::Vector{T}, w::T) where T

    for m in eachindex(g_re)
        g_re[m] *= w
    end
    for m in eachindex(g_im)
        g_im[m] *= w
    end

    return nothing
end

# over m.

function evalSharedShiftgradient!(
    g_re::Vector{T},
    g_im::Vector{T},
    ∂fr::Vector{Vector{T}},
    ∂fi::Vector{Vector{T}},
    c::T,
    s::T,
    ) where T <: AbstractFloat

    for m in eachindex(g_re)

        ∂fr_∂r, ∂fr_∂λ = ∂fr[m]
        ∂fi_∂r, ∂fi_∂λ = ∂fi[m]
        
        g_re[m] += (s*∂fi_∂r - c*∂fr_∂r)
        g_im[m] += -(c*∂fi_∂r + s*∂fr_∂r)
    
    end

    return nothing
end

function evalCoherenceShiftgradient!(
    g_re::Vector{T},
    g_im::Vector{T},
    #frs::Vector{T},
    #fis::Vector{T},
    ∂fr::Vector{Vector{T}},
    ∂fi::Vector{Vector{T}},
    c::T,
    s::T,
    #η::Vector{T},
    η_j::T,
    ) where T <: AbstractFloat

    for m in eachindex(g_re)
        
        #fr = X.re_evals[n][i][k][m]
        #fi = X.im_evals[n][i][k][m]
        
        ∂fr_∂r, ∂fr_∂λ = ∂fr[m]
        ∂fi_∂r, ∂fi_∂λ = ∂fi[m]

        # for j in eachindex(η)
        #     g_re[m] += η[j]*(s*∂fi_∂r - c*∂fr_∂r)#*multiplier
        #     g_im[m] += -η[j]*(c*∂fi_∂r + s*∂fr_∂r)#*multiplier
        # end
        
        g_re[m] += η_j*(s*∂fi_∂r - c*∂fr_∂r)#*multiplier
        g_im[m] += -η_j*(c*∂fi_∂r + s*∂fr_∂r)#*multiplier
    
    end

    return nothing
end

function evalCoherencePhasegradient!(
    g_re::Vector{T},
    g_im::Vector{T},
    frs::Vector{T},
    fis::Vector{T},
    c::T,
    s::T,
    #η::Vector{T},
    η_j::T,
    ) where T <: AbstractFloat

    for m in eachindex(g_re)
        
        fr = frs[m]
        fi = fis[m]
        
        #∂fr_∂r, ∂fr_∂λ = ∂fr[m]
        #∂fi_∂r, ∂fi_∂λ = ∂fi[m]

        # for j in eachindex(η)
        #     g_re[m] += -η[j]*(c*fi + s*fr)#*multiplier
        #     g_im[m] += η[j]*(c*fr - s*fi)#*multiplier
        # end
        g_re[m] += -η_j*(c*fi + s*fr)#*multiplier
        g_im[m] += η_j*(c*fr - s*fi)#*multiplier
        
    end

    return nothing
end


function evalSharedT2gradient!(
    g_re::Vector{T},
    g_im::Vector{T},
    ∂fr::Vector{Vector{T}},
    ∂fi::Vector{Vector{T}},
    c::T,
    s::T,
    λ0::T,
    ) where T <: AbstractFloat

    for m in eachindex(g_re)
        
        ∂fr_∂r, ∂fr_∂λ = ∂fr[m]
        ∂fi_∂r, ∂fi_∂λ = ∂fi[m]

        g_re[m] += (c*∂fr_∂λ - s*∂fi_∂λ)*λ0
        g_im[m] += (c*∂fi_∂λ + s*∂fr_∂λ)*λ0
    end

    return nothing
end


### packages.
"""
```
function packagedevalgradient!(
    model_params::SIG.MixtureModelParameters,
    U_rad,
    x_test::Vector{T},
)::Tuple{Vector{Vector{T}},Vector{Vector{T}}} where T
```

Returns the evaluated gradients.

`x_test` is the parameters to be evaluated at.
`U_rad` is the radial frequency (in radians) locations the objective function uses.
"""
function packagedevalgradient!(
    model_params::SIG.MixtureModelParameters{T,MT},
    U_rad,
    x_test::Vector{T},
    )::Tuple{Vector{Vector{T}},Vector{Vector{T}}} where {T <: AbstractFloat, MT <: SIG.CLMixtureSpinSys}

    mapping = model_params.systems_mapping
    w = model_params.w
    MSS = model_params.MSS

    ## compute gradient using the method under test (batch method).

    model_params.var_flat[:] = x_test
    importmodel!(model_params)

    shift_multi_inds = createmappingLUT(mapping.shift)
    phase_multi_inds = createmappingLUT(mapping.phase)
    T2_multi_inds = createmappingLUT(mapping.T2)

    X = BatchEvalBuffer(U_rad, MSS)
    updategradientbuffer!(X, MSS)
    constructdesignmatrix!(X, MSS)

    gs_re = createbatchgradbuffer(MSS, length(U_rad))
    gs_im = createbatchgradbuffer(MSS, length(U_rad))
    
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

    return gs_re, gs_im
end
    