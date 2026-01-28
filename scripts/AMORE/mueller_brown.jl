using Revise
using ISOKANN
using Plots
using LineSearches
using Optim
using LinearAlgebra
using ForwardDiff
using Random
using Statistics
using Optim
using OrdinaryDiffEq


sim=MuellerBrown(dt = 1e-4, lagtime=1e-3, sigma=5);
traj=ISOKANN.trajectory(sim; T=1000,saveat=1e-1, x0 =[0,0])

data = SimulationData(sim, traj, 10);

#works better than Adam
iso=Iso(data, model=ISOKANN.smallnet(2), opt=NesterovRegularized(1e-3,1e-3), minibatch=1000);
iso=gpu(iso)
run!(iso, 4000);

ISOKANN.plot_training(iso)

chi = vec(cpu(ISOKANN.chis(iso)))
Z = [ISOKANN.mueller_brown(x) for x in eachcol(traj) ]

scatter(traj[1,:], traj[2,:], markerz= chi, xlabel="x", ylabel="y", title="Mueller-Brown \\chi", colorbar_title="\\chi",xlim = (-2,1.3), ylim = (-0.5,3.3),markercolor=:inferno)
scatter(traj[1,:], traj[2,:], markerz= Z, xlabel="x", ylabel="y", title="Mueller-Brown PES", colorbar_title="Energy",xlim = (-2,1.3), ylim = (-0.5,3.3), color=:inferno)
scatter(data.coords[1][1,1:100:end], data.coords[1][2,1:100:end], xlim = (-2,1.3), ylim = (-0.5,3.3))
scatter!(data.coords[2][1,1,1:100:end], data.coords[2][2,1,1:100:end], xlim = (-2,1.3), ylim = (-0.5,3.3))
# Plot


transition,features = transition_state(iso,0.49,0.51)
x0s_samples = transition
scatter!(transition[1,:], transition[2,:],xlim = (-2,1.3), ylim = (-0.5,3.3),color=:orange)

marginal_free_energy(iso)

xs_samples =nothing
for i in 1:size(x0s_samples,2)
    x0 = x0s_samples[:,i]
    xs = reactionpath_minimum(iso,x0;steps=100, f_reltol=1e-5, alphaguess =5e-5, iterations=100,algorithm=ConjugateGradient,show_trace=false)
    xs_samples = isnothing(xs_samples) ? xs : hcat(xs_samples,xs)
end
scatter(traj[1,:], traj[2,:], markerz= chi, xlabel="x", ylabel="y", title="Mueller-Brown \\chi", colorbar_title="\\chi",xlim = (-2,1.3), ylim = (-0.5,3.3),markercolor=:inferno)
scatter(traj[1,:], traj[2,:], markerz= Z, xlabel="x", ylabel="y", title="Mueller-Brown PES", colorbar_title="Energy",xlim = (-2,1.3), ylim = (-0.5,3.3), color=:inferno)

scatter!(x0s_samples[1,:], x0s_samples[2,:],xlim = (-2,1.3), ylim = (-0.5,3.3),color=:orange)
scatter!(xs_samples[1,:], xs_samples[2,:],xlim = (-2,1.3), ylim = (-0.5,3.3),color=:magenta)


x0 = [0;2]
scatter!(x0[1,:], x0[2,:],xlim = (-2,1.3), ylim = (-0.5,3.3),color=:orange)
x = ISOKANN.energyminimization_chilevel(iso,x0)
scatter!(x[1,:], x[2,:],xlim = (-2,1.3), ylim = (-0.5,3.3),color=:magenta)

plotlyjs()
scatter3d(traj[1,:], traj[2,:], chi, xlabel="x", ylabel="y", title="Mueller-Brown \\chi",xlim = (-2,1.3), ylim = (-0.5,3.3))


# ----------------------------- Utilities -----------------------------
# --- cumulative arclength + uniform reparam, robust to numeric issues ---
function reparametrize_uniform!(X::Matrix{Float64})
    M = size(X, 2)
    if M ≤ 2; return X; end

    dx = diff(X; dims=2)                       # 2 x (M-1)
    seg = [norm(dx[:,i]) for i in 1:M-1]       # Vector{Float64}

    if any(!isfinite, seg)
        # fallback: tiny equal spacing to avoid NaNs; leave X as-is
        return X
    end
    s = [0.0; cumsum(seg)]
    L = s[end]
    if L ≤ 0 || !isfinite(L)
        return X
    end
    snew = range(0.0, L; length=M) |> collect

    Y = similar(X)
    for j in 1:M
        sj = snew[j]
        i = clamp(searchsortedlast(s, sj), 1, M-1)
        denom = s[i+1] - s[i]
        t = denom == 0 ? 0.0 : (sj - s[i]) / denom
        @inbounds Y[:,j] = (1-t) .* X[:,i] .+ t .* X[:,i+1]
    end
    X .= Y
    return X
end

# --- perpendicular force (I - ττᵀ)(-∇U) using centered tangent ---
function perp_forces(X::Matrix{Float64}, ∇U::Function)
    M = size(X, 2)
    F = zeros(size(X))
    for j in 2:M-1
        t = X[:,j+1] .- X[:,j-1]
        τ = t / (norm(t) + eps())
        g = ∇U(view(X,:,j))
        f = -g
        F[:,j] = f .- (τ' * f) .* τ   # perpendicular component
    end
    return F
end

# --- light smoothing to stabilize ---
function smooth_string!(X::Matrix{Float64}; α::Float64=0.1)
    M = size(X,2)
    for j in 2:M-1
        @inbounds X[:,j] .= (1-α).*X[:,j] .+ (α/2).*(X[:,j-1] .+ X[:,j+1])
    end
    return X
end

"""
string_mep(U, a, b; M=121, box, dt=0.05, stepmax=0.05, tol=1e-6, maxiter=20000, smooth=true)

Robust simplified string method with:
- step-size control (cap max image move to `stepmax`)
- box clamping after each update
- NaN/Inf guards: backtrack by halving `dt` and retry

Returns (X, idx_sad) where X is 2 x M MEP polyline, idx_sad is highest-energy image.
"""
function string_mep(U::Function, a::Vector{Float64}, b::Vector{Float64};
                    M::Int=121, box::NTuple{4,Float64}=(-2.0, 1.3, -0.5, 3.3),
                    dt::Float64=0.05, stepmax::Float64=0.05,
                    tol::Float64=1e-6, maxiter::Int=20_000, smooth::Bool=true)

    @assert length(a)==2 && length(b)==2
    xmin, xmax, ymin, ymax = box

    # init straight line
    X = hcat([(1-t)*a .+ t*b for t in range(0,1; length=M)]...)
    X[:,1]   .= a
    X[:,end] .= b

    lastmax = Inf
    it = 1
    while it ≤ maxiter
        # compute perpendicular forces
        F = perp_forces(X, x->ForwardDiff.gradient(U, x))
        # step-size control
        Fn = [norm(F[:,j]) for j in 2:M-1]
        Fmax = maximum(Fn; init=0.0)
        # if already converged
        if Fmax < tol && isfinite(Fmax)
            break
        end
        # adaptive dt to cap displacement
        dteff = Fmax > 0 ? min(dt, stepmax / Fmax) : dt

        Xtrial = copy(X)
        Xtrial[:,2:end-1] .+= dteff .* F[:,2:end-1]

        # clamp to box
        @inbounds for j in 2:M-1
            Xtrial[1,j] = clamp(Xtrial[1,j], xmin, xmax)
            Xtrial[2,j] = clamp(Xtrial[2,j], ymin, ymax)
        end

        # reject step if it created non-finite values; backtrack dt
        if any(!isfinite, Xtrial)
            dt *= 0.5
            continue
        end

        # accept
        X .= Xtrial

        # optional smoothing
        if smooth; smooth_string!(X; α=0.1); end

        # reparam (guarded)
        reparametrize_uniform!(X)

        # check energy stagnation
        E = [U(view(X,:,j)) for j in 1:M]
        if any(!isfinite, E)
            dt *= 0.5
            continue
        end
        maxE = maximum(E)
        if abs(lastmax - maxE) < 1e-12 && Fmax < 5tol
            break
        end
        lastmax = maxE
        it += 1
    end

    # pick saddle as highest-energy image
    E = [U(view(X,:,j)) for j in 1:M]
    idx_sad = argmax(E)
    return X, idx_sad
end

# ----- minima  -----
using Optim
function polish_minimum(x0; it=2_000)
    res  = optimize(V, x0, NelderMead(), Optim.Options(iterations=it))
    xnm  = Optim.minimizer(res) |> collect
    res2 = optimize(V, xnm, BFGS(), Optim.Options(iterations=it); autodiff=:forward)
    return Optim.minimizer(res2) |> collect
end

# Example minima
a_guess = [-0.55, 1.45]; b_guess = [0.6, 0.0]          # common MB minima guesses
a = polish_minimum(a_guess)
b = polish_minimum(b_guess)



path1 = [X[:,j] for j in 1:idx_sad]
path2 = [X[:,j] for j in idx_sad:size(X,2)]


using Statistics, ColorSchemes, Plots
default(fontfamily = "Helvetica")
# grid & energies 
box = (-1.2, 1.0, -0.5, 2.0)
nx, ny = 300, 300
xg = range(box[1], box[2], length=nx)
yg = range(box[3], box[4], length=ny)
Zgrid = [V([x,y]) for y in yg, x in xg]  

# robust color limits from percentiles
finiteZ = filter(isfinite, vec(Zgrid))
lo = quantile(finiteZ, 0.0)
hi = quantile(finiteZ, 0.96)


cmap = cgrad(ColorSchemes.roma)   

pPES = contourf(xg, yg, Zgrid;
    levels = range(lo, hi, length=30),
    clims  = (lo, hi),
    c      = cmap,
    xlabel = "x", ylabel = "y",
    #title  = "Müller–Brown PES",
    xlim   = (box[1], box[2]), ylim = (box[3], box[4]),
    colorbar = true, cbar_title = "Energy ",
)
# overlays...


scatter!(xs_samples[1,:], xs_samples[2,:]; ms=6, mc=:blue, label="\\chi - MEP states")


plot!(pPES, X[1,:], X[2,:]; lw=6, lc=:black, label="String MEP",legend=:topright)
scatter!(pPES, [a[1], b[1]], [a[2], b[2]]; ms=6, mc=:cyan, marker=:diamond, label="Minima")
scatter!(x0s_samples[1,:], x0s_samples[2,:]; ms=10, mc=:magenta, marker=:diamond, label="Initial states",
    framestyle=:box,
    size=(1200, 1200),     # Figure size in pixels
    dpi=300,
    xlabelfontsize=26,
    ylabelfontsize=26,
    legendfontsize=24,
    tickfontsize=18, 
    colorbar_titlefontsize=26,
    right_margin=6mm)
savefig("/home/numerik/jkresse/code/CGEM/figures/mueller_pes.png")
#####chis######
pChi = scatter(
    iso.data.coords[1][1,:], iso.data.coords[1][2,:];
    markerz = chi, c = :cividis, ms=4,
    xlabel = "x", ylabel = "y",
    xlim = (box[1], box[2]), ylim = (box[3], box[4]),
    colorbar = true, cbar_title = "Reaction coordinate \\chi ",
    label = "Stationary distribution", framestyle=:box,
    size=(1200, 1200),     # Figure size in pixels
    dpi=300,
    xlabelfontsize=26,
    ylabelfontsize=26,
    legendfontsize=24,
    tickfontsize=18, 
    colorbar_titlefontsize=26,
    right_margin=6mm
)
# --- ∇χ quiver on pChi: uniform subsample across χ, no log scaling ---

coords = iso.data.coords[1]                # 2×N
xs, ys = coords[1,:], coords[2,:]
N = length(xs)

# Gradients at all points
gx = similar(xs); gy = similar(xs)
for i in eachindex(xs)
    g = ISOKANN.dchidx(iso, [xs[i], ys[i]])
    gx[i], gy[i] = g[1], g[2]
end
gmag = sqrt.(gx.^2 .+ gy.^2) .+ 1e-12

# Uniform subsample across χ
nbins, perbin = 20, 5
edges = range(minimum(chi), maximum(chi), length=nbins+1)
sel = Int[]
for b in 1:nbins
    # combine conditions into one BitVector; make last bin closed on the right
    mask = (chi .>= edges[b]) .& (b < nbins ? (chi .< edges[b+1]) : (chi .<= edges[b+1]))
    idx = findall(mask)
    if !isempty(idx)
        k = min(perbin, length(idx))
        take = idx[randperm(length(idx))[1:k]]   # sample without replacement
        append!(sel, take)
    end
end

# Linear scaling of arrow lengths
scale = 0.05 * max(box[2]-box[1], box[4]-box[3]) / maximum(gmag[sel])
ux = scale .* gx
uy = scale .* gy
#plot!(pChi, [NaN], [NaN]; label="∇\\chi", linecolor=:black, lw=1.2)

quiver!(pChi, xs[sel], ys[sel];
        quiver=(ux[sel], uy[sel]),
        linecolor=:black, lw=0.8, arrow=:arrow, arrowsize=0.35,label="∇\\chi", legend=:topright)
# Add a dummy series to put "∇χ" into the legend
savefig("/home/numerik/jkresse/code/CGEM/figures/mueller_chi.png")


ISOKANN.save("/data/numerik/people/jkresse/CGEM/iso_mueller.jld2",iso)


