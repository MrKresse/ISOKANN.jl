using Statistics
using ForwardDiff

"""
     marginal_free_energy(iso::Iso;nbins)

Compute the free energy from the density of chi values.

# Arguments
`iso` the Iso object.
`nbins` the number of bins of the histogram used for estimation.
# Returns
`F` the free energy energy surface of χ in kJ/mol up to an additive constant.
"""
function marginal_free_energy(iso::Iso;nbins=100)
    iso=(cpu(iso))
    # Parameters
    chivals = vec(chis(iso))
  
    sim = iso.data.sim
    kBT = 0.008314463 * OpenMM.temp(sim)
  
    # Create a histogram for the reaction coordinate values
    hist = fit(Histogram, chivals, nbins=nbins)
  
    # Bin centers
    edges = hist.edges[1]
    bin_centers = (edges[1:end-1] + edges[2:end]) ./ 2
  
    # Normalize histogram to get an estimate of the probability density.
    # Note: diff(edges) gives the bin widths.
    P = hist.weights ./ sum(hist.weights .* diff(edges))
  
    # Compute the free energy (up to an additive constant)
    F = -kBT * log.(P)
  
    # Shift free energy so that the minimum is zero (this is arbitrary)
    F .-= minimum(F)
  
    # Plot the free energy profile
    p=plot(bin_centers, F, xlabel="χ", ylabel="Free Energy [kJ/mol]",
        title="Free Energy Profile", legend=false,size=(600,600), frame=:box)
    display(p)
    return(F)
  end
  
  """
    constrained_free_energy(iso, xs; sim, steps)

Compute the free energy using Thermodynamic Integration. 
Starting from the levelset samples xs orhtogonal simulations estimate the 
mean force along χ, which is integrated to yield the PMF. 

# Arguments
`iso` the Iso object.
`xs`  the starting points (which should be well distributed in state space).
`sim` the simulation used for the orthongal sampling.
`steps` the number of steps in each orthogonal simulation.
# Returns
`F` the free energy energy surface of χ in kJ/mol up to an additive constant.
"""
  function constrained_free_energy(iso::Iso, xs; sim::OpenMMSimulation=iso.data.sim, steps=2000)
    iso = cpu(iso)
    n_states = size(xs, 2)
    mean_forces = zeros(n_states)
    mean_Z = zeros(n_states)
  
    dt   = OpenMM.stepsize(sim)
    gamma = OpenMM.friction(sim)
    kBT = 0.008314463 * OpenMM.temp(sim)
    m    = repeat(OpenMM.masses(sim), inner=3)
    chi_vals= [ iso.model(iso.data.featurizer(x))[1] for x in eachcol(xs)]
   
    for i in 1:n_states
        # Initialize state x for the i-th reaction coordinate point.
        x = xs[:, i]
        # Preallocate arrays
        lambdas = zeros(steps)
        Zs = zeros(steps)

        chi_level = chi_vals[i]
        println(chi_level)
  
        v = zeros(length(x))
        for j in 1:steps
            F = OpenMM.force(sim, x)
            dchi = ISOKANN.dchidx(iso, x)
            F_proj = dot(F, dchi) / dot(dchi, dchi)
            #Simulate on the orthogonal (it does not work to project the velocities)
            @. F -= F_proj * dchi
            db = randn(length(x))
            @. v += 1 / m * ((F - gamma * v) * dt + sqrt(2 * gamma * kBT * dt) * db)
            @. x += v * dt
        
            # Correct the position drift (Simulating on the orthogonal is not enough):
            dchi = ISOKANN.dchidx(iso, x)
            phi_val = iso.model(iso.data.featurizer(x))[1]
            error = phi_val - chi_level  
            correction = error / dot(dchi, dchi)
            @. x -= correction * dchi
        
            #Correction for sampling without Fixman Potential 
            # = det(G_M) because dchi is a vector and M⁻1 is a Diagonal matrix with 1/m_i 
            Z = sum(1 ./ m .* dchi.^2)
            Zs[j] = Z
            #Force along chi (Just the Hamiltonian component converges to F too)
            lambdas[j] = -F_proj
        end
        println(iso.model(iso.data.featurizer(x))[1])
  
        # Compute mean force
        mean_forces[i] = mean(lambdas)
        mean_Z[i] = mean(1 ./sqrt.(Zs))
    end
    #sort by chi
    inds= sortperm(chi_vals)
    mean_forces=mean_forces[inds]
    chi_vals=chi_vals[inds]
    mean_Z = mean_Z[inds]
  
    #F_rgd = reverse(integrate_chi(reverse(mean_forces),reverse(chi_vals)))
    F_rgd = integrate_chi(mean_forces,chi_vals)
    F_std =  F_rgd .- kBT*log.(mean_Z)
    p =plot(chi_vals, F_std, xlabel="χ", ylabel="PMF [kj/mol]", legend=false, size=(600,600), frame=:box)
    display(p)
    return F_std
  end
  
  """
  local_mean_force(iso, xs; sim, steps)

Compute the free energy using Thermodynamic Integration. 
Bins the samples into levelsets, computes the mean force along χ locally in
every levelset. (Extremely extensive sampling necessary.)

# Arguments
`iso` the Iso object.
`xs`  the starting points (which should be well distributed in state space).
`nbins` The number of bins/levelsets.
# Returns
`F` the free energy surface of χ in kJ/mol up to an additive constant.
"""
  function local_mean_force(iso::Iso,xs,nbins)
    sim =iso.data.sim
    chi_vals= [ iso.model(iso.data.featurizer(x))[1] for x in eachcol(xs)]
    num_states= size(xs,2)
    inds= sortperm(chi_vals)
    chi_vals = chi_vals[inds]
    # Reorder the columns of xs according to sorted indices
    xs_sorted = xs[:, inds]
  
    bin_size = div(num_states, nbins)
    
    # Preallocate an array to hold the bins (each bin is a submatrix of xs)
    bins = Vector{Matrix{eltype(xs)}}(undef, nbins)
    
    mean_chi_vals=zeros(nbins)
    r = mod(num_states, nbins)
    start_idx = 1
    for i in 1:nbins
      extra = i <= r ? 1 : 0
      end_idx = start_idx + bin_size + extra - 1
      bins[i] = xs_sorted[:, start_idx:end_idx]
      mean_chi_vals[i] = mean(chi_vals[start_idx:end_idx])
      start_idx = end_idx + 1
    end
  
  
    mean_forces = zeros(nbins)
    #mean_Z = zeros(n_bins)
    for i in 1:nbins
      current_bin_size = size(bins[i],2)
      lambdas = zeros(current_bin_size)
      for j in 1:bin_size
        x = bins[i][:,j]
        F = OpenMM.force(sim, x)
        dchi = ISOKANN.dchidx(iso, x)
        F_proj = dot(F, dchi) / dot(dchi, dchi)
        #Correction for sampling without Fixman Potential (do i need this here?)
        # = det(G_M) because dchi is a vector and M⁻1 is a Diagonal matrix with 1/m_i 
        #Z = sum(1 ./ m .* dchi.^2)
        #Zs[j] = Z
  
        #Force along chi (Just the Hamiltonian component converges to F too)
        lambdas[j] = -F_proj
      end
      mean_forces[i]=mean(lambdas)
      #mean_Z[i]=mean(Zs)
    end
  
  
    F_rgd = integrate_chi(mean_forces,mean_chi_vals)
    #F_std =  F_rgd .- kBT*log.(mean_Z)
    p =plot(mean_chi_vals, F_rgd, xlabel="χ", ylabel="PMF [kj/mol]", legend=false, size=(600,600), frame=:box)
    display(p)
    return F_rgd
  end

  """
  integrate_chi(f, chi_vals)

Cumulative integral of the mean force with respect to χ using the trapezoid rule.

# Arguments
`f` The mean force.
`chi_vals` The levelset χ values.
# Returns
`F` the (rigid) free energy surface of χ.
"""
 function integrate_chi(f, chi_vals)
    n = length(chi_vals)
    F = zeros(n)
    # Use the trapezoidal rule; F[1] is set to zero as reference.
    for i in 2:n
        dχ = chi_vals[i] - chi_vals[i-1]
        F[i] = F[i-1] + 0.5 * (f[i] + f[i-1]) * dχ
    end
    return F
end

    """
    delta_G(PMF,chi_vals)
Convenience function to compute free energy differences in a double well free energy surface.
"""
function delta_G(PMF,chi_vals)
    chi_vals = sort(chi_vals)
   
    G0 = minimum(PMF[chi_vals.<0.5])
    G1 = minimum(PMF[chi_vals.>=0.5])
    return G0-G1
end

    """
    function sample_coords(iso,n_points;xs)
Convenience function to uniformly sample npoints out of the χ distribution of xs coordinates.
"""
function sample_coords(iso::Iso,n_points;xs=hcat(iso.data.coords[1],iso.data.coords[2][:,:,1]))
    chi_vals = cpu([iso.model(iso.data.featurizer(x))[1] for x in eachcol(xs)])
    # Determine the range of χ values.
    chi_min, chi_max = minimum(chi_vals), maximum(chi_vals)
    # Create n_points uniformly spaced target χ values.
    target_chis = range(chi_min, stop=chi_max, length=n_points)
    
    # For each target χ, find the index in chi_vals with the minimum absolute difference.
    indices = map(t -> argmin(abs.(chi_vals .- t)), target_chis)
    
    # Extract the corresponding columns from xs.
    sampled_coords = xs[:, indices]
    return sampled_coords
end
  
# ================= OPES 1D (periodic + Neff bandwidth + normalized KDE + WT + barrier ΔE) ===================

mutable struct OPES1D
    beta::AbstractFloat
    stride::Int
    sigma::AbstractFloat
    Vcap::AbstractFloat
    centers::Vector{AbstractFloat}
    stepcount::Int
    flat_lo::AbstractFloat
    flat_hi::AbstractFloat
    warmup::Int
    periodic::Bool
    period::AbstractFloat

    # Neff-based σ
    adapt_sigma::Bool
    sigma0::AbstractFloat
    w_sum::AbstractFloat
    w2_sum::AbstractFloat

    # monitoring
    sigma_factor::AbstractFloat
    μ::AbstractFloat
    m2::AbstractFloat
    nobs::Int

    # WT + barrier
    wt::Bool                # enable WT scaling DOES NOT WORK YET
    gamma::AbstractFloat          # bias factor (>1 typically); may be set independently
    ΔE::AbstractFloat             # barrier parameter (kJ/mol)
    eps_reg::AbstractFloat        # ε derived from (β, γ, ΔE)
end

# --- internal: derive ε from (β, γ, ΔE) 
@inline function _derive_eps(beta::AbstractFloat, gamma::AbstractFloat, ΔE::AbstractFloat)
    # ε = e^{-βΔE} / (1 - 1/γ)
    # guard against γ≈1
    @assert gamma > 1.0 "gamma must be > 1.0 to define ε from ΔE (set wt=false for flat-OPES)."
    return exp(-beta * ΔE) / (1 - 1/gamma)
end

function OPES1D(; beta, stride=500, sigma, Vcap=75.0,
                 flat_lo=-Inf, flat_hi=Inf, warmup=0,
                 periodic=false, period=2π,
                 adapt_sigma=true, sigma_factor=1/10,
                 # WT+barrier params
                 wt=true, ΔE=50.0, gamma=nothing)

    # If gamma not provided, link it to ΔE: γ = β ΔE
    γ = isnothing(gamma) ? (beta * ΔE) : gamma
    @assert γ ≥ 1.0 "gamma must be ≥ 1.0 (set >1 for WT; =1 is flat-OPES scaling)."

    opes = OPES1D(beta, stride, sigma, Vcap, AbstractFloat[], 0,
                  flat_lo, flat_hi, warmup, periodic, period,
                  adapt_sigma, sigma, 1.0, 1.0,
                  sigma_factor, 0.0, 0.0, 0,
                  wt, γ, ΔE, _derive_eps(beta, γ, ΔE))
    return opes
end

# ---------- logging ----------
mutable struct OPESLog
    xis::Vector{AbstractFloat}
    Vs::Vector{AbstractFloat}
    saveevery::Int
    step::Int
end

# ================ internals ================

@inline function delta_cv(opes::OPES1D, ξ::AbstractFloat, c::AbstractFloat)
    if !opes.periodic
        return ξ - c
    else
        p = opes.period
        return mod(ξ - c + p/2, p) - p/2
    end
end

# Properly normalized log-KDE of the centers with common σ:
# log p̂(ξ) = log( (1/M) * Σ_k ϕσ(ξ - c_k) ),  ϕσ(u)=(1/(σ√(2π)))exp(-u²/(2σ²))
function logkde(opes::OPES1D, ξ::AbstractFloat)
    M = length(opes.centers)
    M == 0 && return -Inf
    invσ = 1/opes.sigma
    invσ2 = invσ^2
    maxa = -Inf
    @inbounds for c in opes.centers
        d = delta_cv(opes, ξ, c)
        a = -0.5 * d*d * invσ2
        maxa = (a > maxa) ? a : maxa
    end
    s = 0.0
    @inbounds for c in opes.centers
        d = delta_cv(opes, ξ, c)
        s += exp(-0.5 * d*d * invσ2 - maxa)
    end
    lognorm = -log(M) - log(opes.sigma) - 0.5*log(2π)
    return log(s + eps()) + maxa + lognorm
end

# ∂ξ log p̂(ξ) (constants cancel)
function dlogkde_dξ(opes::OPES1D, ξ::AbstractFloat)
    M = length(opes.centers)
    M == 0 && return 0.0
    invσ2 = 1/(opes.sigma^2)
    num = 0.0; den = 0.0
    @inbounds for c in opes.centers
        d = delta_cv(opes, ξ, c)
        w = exp(-0.5 * d^2 * invσ2)
        den += w
        num += w * (-d) * invσ2
    end
    return den > 0 ? num/den : 0.0
end

# Bias with WT + barrier:
# V = (1/(βγ)) log( p̂ + ε ). With ε from ΔE ⇒ min V = -ΔE when p̂→0.
# Clamp V to ±Vcap and zero force if clamped.
function bias_and_grad_ξ(opes::OPES1D, ξ::AbstractFloat)
    ℓp  = logkde(opes, ξ)
    dℓp = dlogkde_dξ(opes, ξ)

    if opes.wt
        α = 1.0 / (opes.beta * opes.gamma)
        # log(p̂ + ε) = logaddexp(log p̂, log ε)
        V_unc = α * log(exp(ℓp) + opes.eps_reg)
        V     = clamp(V_unc, -opes.Vcap, opes.Vcap)
        # d/dξ log(p̂ + ε) = (p̂' / (p̂ + ε))
        dlog = dℓp * (exp(ℓp) / (exp(ℓp) + opes.eps_reg))
        dVdξ = (V == V_unc) ? α * dlog : 0.0
        return V, dVdξ
    else
        # flat target (OPES-E): V = (1/β) log p̂
        V_unc = (1/opes.beta) * ℓp
        V     = clamp(V_unc, -opes.Vcap, opes.Vcap)
        dVdξ  = (V == V_unc) ? (1/opes.beta) * dℓp : 0.0
        return V, dVdξ
    end
end

@inline function _welford_update!(opes::OPES1D, ξ::AbstractFloat)
    opes.nobs += 1
    δ = ξ - opes.μ
    opes.μ += δ / opes.nobs
    opes.m2 += δ * (ξ - opes.μ)
end

# Neff σ schedule (d=1)
@inline function _update_sigma_neff!(opes::OPES1D)
    Neff = (opes.w_sum^2) / opes.w2_sum
    scale = (Neff * 3 / 4) ^ (-1/5)                  # (d+2)/4 with d=1
    opes.sigma = max(opes.sigma0 * scale, 1e-6)
end

# Lightweight optional compression (disabled by default)
function maybe_compress!(opes::OPES1D; dt=0.0)
    dt <= 0 && return
    σ = opes.sigma
    newc = opes.centers[end]
    if length(opes.centers) >= 2
        idx = argmin(abs.(opes.centers[1:end-1] .- newc) ./ σ)
        dmin = abs(opes.centers[idx] - newc) / σ
        if dmin < dt
            opes.centers[idx] = 0.5*(opes.centers[idx] + newc)
            pop!(opes.centers)
        end
    end
end

function maybe_deposit!(opes::OPES1D, ξ::AbstractFloat)
    opes.stepcount += 1
    _welford_update!(opes, ξ)

    if opes.stepcount % opes.stride == 0
        push!(opes.centers, ξ)
        opes.w_sum  += 1.0
        opes.w2_sum += 1.0
        maybe_compress!(opes; dt=0.0)
        if opes.adapt_sigma
            _update_sigma_neff!(opes)
        end
    end

    if opes.adapt_sigma && opes.warmup > 0 && opes.stepcount >= opes.warmup
        opes.adapt_sigma = false
    end
    return nothing
end

# ================ closures ================

function opes_bias_closure_with_log(opes::OPES1D; xi::Function, dxi_dr::Function, log::OPESLog)
    return function B(q; t=nothing, sigma=nothing, F=nothing)
        ξ = xi(q)
        maybe_deposit!(opes, ξ)
        V, dVdξ = bias_and_grad_ξ(opes, ξ)

        log.step += 1
        if log.step % log.saveevery == 0
            push!(log.xis, ξ)
            push!(log.Vs,  V)
        end

        if opes.stepcount < opes.warmup
            return zeros(eltype(q), size(q))
        end
        g = dxi_dr(q)
        return @. -dVdξ * g
    end
end

function opes_bias_closure(opes::OPES1D; xi::Function, dxi_dr::Function)
    return function B(q; t=nothing, sigma=nothing, F=nothing)
        ξ = xi(q)
        maybe_deposit!(opes, ξ)
        if opes.stepcount < opes.warmup
            return zeros(eltype(q), size(q))
        end
        _, dVdξ = bias_and_grad_ξ(opes, ξ)
        g = dxi_dr(q)
        return @. -dVdξ * g
    end
end

# ================ helpers ================

function estimate_sigma_phi(sim; steps=50_000, saveevery=100)
    xs = ISOKANN.trajectory(sim, steps; saveevery=saveevery)
    φs = [ISOKANN.phi(xs[:, i]) for i in 1:size(xs,2)]
    return std(φs) / 10
end

function periodic_weighted_kde(xs::AbstractVector, ws::AbstractVector, grid::AbstractVector, bw::AbstractFloat)
    y = zeros(length(grid))
    inv2σ2 = 1.0/(2*bw^2)
    inv_norm = 1/(bw*sqrt(2π))
    @inbounds for (xi, wi) in zip(xs, ws)
        @inbounds for m in -1:1
            xc = xi + m*2π
            @. y += wi * inv_norm * exp(- (grid - xc)^2 * inv2σ2)
        end
    end
    dx = grid[2] - grid[1]
    s = sum(y) * dx
    return s > 0 ? y ./ s : y
end

@inline wrapdiff(φa, φb) = mod(φa - φb + π, 2π) - π

function dphi_dx_periodic_FD(x::AbstractVector; idxs=nothing, h=1e-6)
    N = length(x)
    g = zeros(eltype(x), N)
    idxs === nothing && (idxs = 1:N)
    for i in idxs
        xp = copy(x); xm = copy(x)
        xp[i] += h;   xm[i] -= h
        φp = phi_val(xp)
        φm = phi_val(xm)
        g[i] = wrapdiff(φp, φm) / (2h)
    end
    return g
end

function dphi_dx_periodic_AD(x::AbstractVector)
    φ0 = phi_val(x)
    ∇c = ForwardDiff.gradient(y -> cos(phi_val(y)), x)
    ∇s = ForwardDiff.gradient(y -> sin(phi_val(y)), x)
    return @. -sin(φ0) * ∇c + cos(φ0) * ∇s
end

# -------- convenience setters (recompute ε when ΔE or γ change) --------
function set_barrier!(opes::OPES1D; ΔE=nothing, gamma=nothing)
    if ΔE !== nothing
        opes.ΔE = ΔE
        if gamma === nothing && opes.wt
            opes.gamma = opes.beta * ΔE    # tie γ to ΔE if user didn't override
        end
    end
    if gamma !== nothing
        @assert gamma ≥ 1.0
        opes.gamma = gamma
    end
    if opes.wt
        @assert opes.gamma > 1.0 "gamma must be > 1 to use a ΔE-derived ε."
        opes.eps_reg = _derive_eps(opes.beta, opes.gamma, opes.ΔE)
    end
    return opes
end