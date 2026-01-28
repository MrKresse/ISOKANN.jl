using Revise
using ISOKANN
using ForwardDiff
using Plots
using StatsPlots
using StatsBase
using KernelDensity
using JLD2   # for saving/loading ref trajs if needed
using Optim

# ----------------- user CV and gradient -------------------------------------
phi_val(x) = ISOKANN.phi(x)
dphi_dx(x) = dphi_dx_periodic_AD(x)

# ----------------- sim + thermodynamics -------------------------------------
sim = ISOKANN.OpenMMSimulation()

kB  = 0.008314463          # kJ / mol / K
T   = OpenMM.temp(sim)
β   = 1.0 / (0.008314463 * T)   # 1/(kB T) in mol/kJ
σ0  = 0.15                          # rad, e.g. from a short unbiased run
ΔE  = 50.0                          # kJ/mol  
gamma = 40.0
# Link γ to ΔE (γ = βΔE) and derive ε from (β,γ,ΔE)
opes = OPES1D(beta=β, sigma=σ0; ΔE=ΔE, wt=false, gamma=gamma)  # gamma defaults to βΔE if wt = true and wt=false is flat target
saveevery = 500
logs = OPESLog(Float64[], Float64[], saveevery, 0)

B = opes_bias_closure_with_log(opes; xi=phi_val, dxi_dr=dphi_dx, log=logs)

# ----------------- run biased dynamics  ------
nsteps = 10_000_000   # total integrator steps 
@time ws = OpenMM.langevin_girsanov!(sim, nsteps; bias=B, saveevery=saveevery, showprogress=true)

#ws = load("/data/numerik/people/jkresse/enhanced_sampling/ad_vacuum_opes_phi_20ns_girs.jld2")["ws"]
xs = ws.values

p_scatter = scatter_ramachandran(xs; title="Biased trajectory Ramachandran")

# ----------------- gather reweighting / ESS ---------------------------------
φ_vals = mod.(logs.xis .+ π, 2π) .- π
V_vals = logs.Vs
w = exp.(β .* V_vals)
ESS = (sum(w)^2) / sum(w.^2)
@info "Saved frames = $(length(φ_vals)); ESS = $ESS; ESS/N = $(ESS/length(φ_vals))"


# ----------------- histogram-based PMF -------------------------------------
nbins = 72
edges = range(-π, stop=π, length=nbins+1)
h = fit(Histogram, φ_vals, Weights(w), edges)
Pφ = h.weights ./ sum(h.weights)   # normalized probability per bin
centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])

Fφ = @. -(1/β) * log(max(Pφ, 1e-16))
Fφ .-= minimum(Fφ)

p1 = plot(centers, Fφ, lw=2, xlabel="φ [rad]", ylabel="PMF(φ) [kJ/mol]",
          label="OPES reweighted (hist)", title="Alanine dipeptide — φ PMF")




σφ = std(φ_vals)
bw = max(σφ/8, 0.02)
grid = range(-π, stop=π, length=360)
p_grid = periodic_weighted_kde(φ_vals, w, collect(grid), bw)
F_kde = @. -(1/β) * log(max(p_grid, 1e-300))
F_kde .-= minimum(F_kde)

plot!(p1, grid, F_kde, lw=2, label="OPES reweighted (KDE)", linestyle=:dash)
#savefig("/home/numerik/jkresse/code/enhanced/PMF_ad")


P = exp.(-β .* F_kde)
P ./= sum(P)   # normalize to 1

# also plot probability
p2 = plot(grid, P, lw=2, xlabel="φ [rad]", ylabel="P(φ)",
          label="OPES reweighted P(φ)")
p = plot(centers, Pφ, xlabel="φ [rad]", ylabel="P(φ)")

# ----------------- ramachandran scatter (biased raw samples) ----------------

@save "/data/numerik/people/jkresse/enhanced_sampling/ad_vacuum_opes_phi_20ns_log.jld2" logs 
@save "/data/numerik/people/jkresse/enhanced_sampling/ad_vacuum_opes_phi_20ns_girs.jld2" ws 

ws = load("/data/numerik/people/jkresse/enhanced_sampling/ad_vacuum_opes_phi_20ns_girs.jld2")["ws"]

# ----------------- display/save figures ------------------------------------
plot(p1)     # PMF (hist + KDE)
plot(p2)     # probability distribution
display(p_scatter)

path_data = "/data/numerik/people/jkresse/CGEM/"
#iso=ISOKANN.load("/scratch/htc/jkresse/transfer/iso_adaptive.jld2")

include("/home/numerik/jkresse/code/bachelor/visualization.jl")
sim = OpenMMSimulation(;steps=100)

intpairs = Vector{Tuple{Int, Int}}()

for i in 1:22
  for j in i+1:22
      push!(intpairs, (i, j))
  end
end
featurizer = OpenMM.FeaturesPairs(intpairs)

#######################unbiased burst samples##############################
#the initial sampling does not have to come from unbiased simulations
data = ISOKANN.SimulationData(sim,xs,1;featurizer=featurizer)



iso = Iso(data,opt=NesterovRegularized(),minibatch = 1000)
iso =gpu(iso)
run!(iso,10000)

ISOKANN.plot_training(iso)
scatter_ramachandran(iso)
ISOKANN.save(path_data*"iso_ad_opes.jld2", iso)
iso = ISOKANN.load(path_data*"iso_ad_opes.jld2")
iso = gpu(iso)
#################Extracting the minimum path################################
traj = iso.data.coords[1]
xss=nothing
 
transition= transition_state(cpu(iso),0.46,0.54)[1]
lo, hi = 0.49, 0.51
step   = 0.01


for i in 1:10
    transition = transition_state(cpu(iso), lo, hi)[1]

    while size(transition, 2) == 0
        lo -= step
        hi += step
        transition = transition_state(cpu(iso), lo, hi)[1]
    end

    j = rand(1:size(transition, 2))
    x = transition[:,j]
    xs = reactionpath_minimum(iso, x;steps=100,f_reltol=1e-5, alphaguess =1e-5, iterations=250,show_trace=false, algorithm = Optim.ConjugateGradient)
    xss = isnothing(xss) ? xs : hcat(xss,xs)

    phi = [ISOKANN.phi(x) for x in eachcol(transition)]
    psi = [ISOKANN.psi(x) for x in eachcol(transition)]

    phi_path = [ISOKANN.phi(x) for x in eachcol(xss)]
    psi_path = [ISOKANN.psi(x)  for x in eachcol(xss)]

    vals   = Float64.([-π, -π/2, 0, π/2, π])
    labels = ["-π", "-π/2", "0", "π/2", "π"]
    tickspec = (vals, labels)


    p = scatter(ISOKANN.phi(traj)[1:10:end], ISOKANN.psi(traj)[1:10:end],
        #markerz=[iso.model(iso.data.featurizer(x))[1] for x in eachcol(traj)],
        markercolor=:grey,
        label="Biased distribution",
        framestyle=:box,
        size=(1200, 1200),
        dpi=300,
        markersize=3,
        markerstrokewidth=0,
        markeralpha=1,
        xlabel="φ [rad]",
        ylabel="ψ [rad]",
        xlabelfontsize=26,
        ylabelfontsize=26,
        legendfontsize=26,
        tickfontsize=18,
        xlim=(vals[1], vals[end]),
        ylim=(vals[1], vals[end]),
        xticks=tickspec,
        yticks=tickspec
    )

    scatter!(phi,psi;
        marker=:diamond,
        markercolor=:magenta,
        markersize=6,
        label="Initial states",
        title="", markerstrokewidth=0,
        markeralpha=1)

    scatter!(phi_path, psi_path;
        markercolor=:blue,
        label="\\chi - MEP states",
        markersize=3, title="", 
        markerstrokewidth=0,markeralpha=1,)
    plot!(legend=:topright)


    savefig("/home/numerik/jkresse/code/CGEM/figures/ad_opes.png")

    sim = OpenMMSimulation()
    newdata = SimulationData(sim, xs, 1)
    data = ISOKANN.mergedata(iso.data, newdata)
    iso.data = data 
    iso =gpu(iso)
    run!(iso,10000)
end
ISOKANN.save(path_data*"iso_ad_opes_enhanced.jld2", iso)
@save path_data*"ad_all_paths_opes.jld2" xss

using Plots
using Colors

# ---- color scheme  ----
c_iter1  = colorant"#D55E00"   # vermillion (distinct from orange)
c_iter5 = colorant"#009E73"   # teal
c_iter10  = :blue       

iters = [1, 5, 10]
cols  = [c_iter1, c_iter5, c_iter10]

vals   = Float64.([-π, -π/2, 0, π/2, π])
labels = ["-π", "-π/2", "0", "π/2", "π"]
 tickspec = (vals, labels)
# Optional styling to make progression visible without reading legend
ms    = [3, 4, 5]           # marker sizes
ma    = [0.55, 0.85, 1.0]# marker alpha
lw    = [1.5, 2.5, 3.0]   # line width (if you also want lines)
ls    = [:dash, :solid, :solid]

# ---- base scatter: sampling (traj) ----
p = scatter(
    ISOKANN.phi(traj), ISOKANN.psi(traj);
    markercolor = :grey,
    label = "OPES of \\Psi biased samples",   
    framestyle = :box,
    size = (1200, 1200),
    dpi = 300,
    markersize = 3,
    markerstrokewidth = 0,
    markeralpha = 0.25,              # <- grey cloud reads better with lower alpha
    xlabel = "φ [rad]",
    ylabel = "ψ [rad]",
    xlabelfontsize = 26,
    ylabelfontsize = 26,
    legendfontsize = 26,
    tickfontsize = 18,
    xlim = (vals[1], vals[end]),
    ylim = (vals[1], vals[end]),
    xticks = tickspec,
    yticks = tickspec,
)

# ---- overlay: selected χ-MEP iterations from xss blocks (100 cols each) ----
for (j, (it, col)) in enumerate(zip(iters, cols))
    r = (it - 1) * 100 + 1 : it * 100
    xs_it = @view xss[:, r]

    ϕ = ISOKANN.phi(xs_it)
    ψ = ISOKANN.psi(xs_it)

    # plot as points (states)
    scatter!(
        ϕ, ψ;
        markercolor = col,
        markersize = ms[j],
        markeralpha = ma[j],
        markerstrokewidth = 0,
        label = "\\chi-MEP (Iteration $(it))",
        title = "",
    )
end

plot!(p; legend = :topright)
savefig("/home/numerik/jkresse/code/CGEM/figures/ad_opes_iters.png")



ISOKANN.savecoords(path_data*"/minimum_path_ad_long.pdb",iso,xs[:,1:100])

xs = ISOKANN.load_trajectory(path_data*"/minimum_path_ad_long.pdb")

#xx = hcat(x,x)#workaround to be able to save just one frame
#ISOKANN.savecoords(path_data*"/start_point_dialanine.pdb",iso,xx)

################Visualizing the gradient###################################
iso = cpu(iso)
gradc= averageGradient(iso;xs=iso.data.coords[1],nbins=100)

gradients=gradc
@save path_data*"/norm_gradient_ad.jld2" gradients
savefig("/home/numerik/jkresse/code/CGEM/figures/heatmap_ad_long.png")

rankAtomImportance(gradc)

gradc= coordGradient(iso,xs)

sumHeatmap(gradc)
gradf= featureGradient(iso,xs)
featureHeatmap(gradf)
saveSum(gradc;out=path_data*"/norm_gradient_path_long.jld2")

intpairs[argmin(sum(gradf,dims=2))]
intpairs[argmax(sum(gradf,dims=2))]
#start pymol:
#run /home/numerik/jkresse/code/bachelor/pymolgradient.py
#path_pdb = "/data/numerik/people/jkresse/CGEM/minimum_path_ad.pdb"
#path_gradient = "/data/numerik/people/jkresse/CGEM/norm_gradient_ad.jld2"
#showFramewiseGradient(path_gradient, path_pdb)



