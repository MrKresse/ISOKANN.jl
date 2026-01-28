using Revise
using ISOKANN
using ForwardDiff
using Plots
using StatsPlots
using StatsBase
using KernelDensity
using JLD2   # for saving/loading ref trajs if needed
using Optim
path_data = "/data/numerik/people/jkresse/CGEM/"

# ----------------- user CV and gradient -------------------------------------
#chi_val(x) = iso.model(iso.data.featurizer(x))[1]
chi_val(x) = cpu(ISOKANN.chicoords(iso,x))[1]
dchi_dx(x) = ISOKANN.dchidx(iso,x)

iso = iso = ISOKANN.load(path_data*"iso_ad_long.jld2")#5 us trained chi
iso = gpu(iso)
# ----------------- sim + thermodynamics -------------------------------------
sim = ISOKANN.OpenMMSimulation()

kB  = 0.008314463          # kJ / mol / K
T   = OpenMM.temp(sim)
β   = 1.0 / (0.008314463 * T)   # 1/(kB T) in mol/kJ
sigma_0 = Statistics.std(chicoords(iso, iso.data.coords[1]))
ΔE  = 50.0                          # kJ/mol  
gamma = 40.0
# Link γ to ΔE (γ = βΔE) and derive ε from (β,γ,ΔE)
opes = OPES1D(beta=β, sigma=sigma_0; ΔE=ΔE, wt=false, gamma=gamma)  # gamma defaults to βΔE if wt = true and wt=false is flat target
saveevery = 50
logs = OPESLog(Float64[], Float64[], saveevery, 0)

B = opes_bias_closure_with_log(opes; xi=chi_val, dxi_dr=dchi_dx, log=logs)

# ----------------- run biased dynamics  ------
nsteps = 1_000_000   # total integrator steps 
@time ws = OpenMM.langevin_girsanov!(sim, nsteps; bias=B, saveevery=saveevery, showprogress=true)

#ws = load("/data/numerik/people/jkresse/enhanced_sampling/ad_vacuum_opes_phi_20ns_girs.jld2")["ws"]
xs = ws.values

p_scatter = scatter_ramachandran(xs; title="Biased trajectory Ramachandran")
savefig("/home/numerik/jkresse/code/CGEM/figures/opes_of_chi.png")


# ----------------- gather reweighting / ESS ---------------------------------
φ_vals = mod.(logs.xis .+ π, 2π) .- π
V_vals = logs.Vs
w = exp.(β .* V_vals)
ESS = (sum(w)^2) / sum(w.^2)
@info "Saved frames = $(length(φ_vals)); ESS = $ESS; ESS/N = $(ESS/length(φ_vals))"



@save "/data/numerik/people/jkresse/enhanced_sampling/ad_vacuum_opes_chi_2ns_log.jld2" logs 
@save "/data/numerik/people/jkresse/enhanced_sampling/ad_vacuum_opes_chi_2ns_girs.jld2" ws 

ws = load("/data/numerik/people/jkresse/enhanced_sampling/ad_vacuum_opes_chi_2ns_girs.jld2")["ws"]

##############################################################################################
include("/home/numerik/jkresse/code/bachelor/visualization.jl")
using Measures
traj = xs
chi_vals = logs.xis
vals   = Float64.([-π, -π/2, 0, π/2, π])
labels = ["-π", "-π/2", "0", "π/2", "π"]
 tickspec = (vals, labels)

# ---- base scatter: sampling (traj) ----
p = scatter(
    ISOKANN.phi(traj), ISOKANN.psi(traj);
    markerz = chi_vals, 
    c = :cividis,
    label = "OPES of \\chi biased samples",   
    framestyle = :box,
    size = (1200, 1200),
    dpi = 300,
    markersize = 5,
    markeralpha = 0.7,
    markerstrokewidth = 0,
    xlabel = "φ [rad]",
    ylabel = "ψ [rad]",
    xlabelfontsize = 26,
    ylabelfontsize = 26,
    legendfontsize = 24,
    tickfontsize = 18,
    xlim = (vals[1], vals[end]),
    ylim = (vals[1], vals[end]),
    xticks = tickspec,
    yticks = tickspec,
    colorbar = true, 
    cbar_title = "Reaction coordinate \\chi ",
    colorbar_titlefontsize=26,
    right_margin=6mm,
    cbar_tickfontsize = 14
)


plot!(p; legend = :topright)
savefig("/home/numerik/jkresse/code/CGEM/figures/ad_opes_of_chi.png")
