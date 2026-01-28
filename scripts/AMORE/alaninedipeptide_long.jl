using Revise
using ISOKANN
using MultivariateStats
using Clustering
using Distances
using Optim
using JLD2
using Random

path_data = "/data/numerik/people/jkresse/CGEM/"
#iso=ISOKANN.load("/scratch/htc/jkresse/transfer/iso_adaptive.jld2")

include("/home/numerik/jkresse/code/bachelor/visualization.jl")
sim = OpenMMSimulation()

intpairs = Vector{Tuple{Int, Int}}()

for i in 1:22
  for j in i+1:22
      push!(intpairs, (i, j))
  end
end
featurizer = OpenMM.FeaturesPairs(intpairs)

#######################long unbiased simulation###############################
traj = OpenMM.trajectory(sim,100_000_000;saveevery = 10000)

rng = MersenneTwister(42)   # reproducible
nframes = size(traj, 2)

frames = randperm(rng, nframes)[1:10]
init = traj[:,frames]

steps = 250_000_000
saveevery = 250
for i in 1:10
    x = init[:,i]
    OpenMM.setcoords(sim,x)

    traj = OpenMM.trajectory(sim,steps; saveevery=saveevery)
    phi = ISOKANN.phi(traj)
    psi = ISOKANN.psi(traj)
    subtraj = traj[:,1:100:end]
    @save "/scratch/htc/jkresse/ad/500ns_$i.jld2" subtraj
    @save "/scratch/htc/jkresse/ad/500ns_phi_$i.jld2" phi
    @save "/scratch/htc/jkresse/ad/500ns_psi_$i.jld2" psi
end

phi = nothing 
psi = nothing 
traj = nothing
tuples = nothing
data = nothing

for i in 1:10
    subtraj = load("/scratch/htc/jkresse/ad/500ns_$i.jld2")["subtraj"]
    traj = isnothing(traj) ? subtraj : hcat(traj, subtraj)
    subphi = load("/scratch/htc/jkresse/ad/500ns_phi_$i.jld2")["phi"]
    phi = isnothing(phi) ? subphi : vcat(phi, subphi)
    subpsi = load("/scratch/htc/jkresse/ad/500ns_psi_$i.jld2")["psi"]

    psi = isnothing(psi) ? subpsi : vcat(psi, subpsi)
    tuples_replica = data_from_trajectory(subtraj;lag = 10)
    subdata = SimulationData(sim,tuples_replica; featurizer=featurizer)#no intra replica pairs
    data =  isnothing(data) ? subdata : ISOKANN.mergedata(data, subdata)
end

#scatter_ramachandran(traj)

#sim = OpenMMSimulation(;steps = 2500)
#data = SimulationData(sim, traj[:,1:10:end], 1;featurizer=featurizer)
#tuples = data_from_trajectory(traj;lag = 10)
#data = SimulationData(sim,tuples; featurizer=featurizer)

iso = Iso(data,opt=NesterovRegularized(),minibatch = 10000)
iso =gpu(iso)
run!(iso,2000)

ISOKANN.plot_training(iso)
savefig("/home/numerik/jkresse/code/CGEM/figures/ad_long_loss.png")

scatter_ramachandran(iso)
savefig("/home/numerik/jkresse/code/CGEM/figures/ad_long_ramachandran.png")

ISOKANN.save(path_data*"iso_ad_long.jld2", iso)
iso = ISOKANN.load(path_data*"iso_ad_long.jld2")

#################Extracting the minimum path################################
traj = iso.data.coords[1]
xss=nothing
for i in 1:10
    transition= transition_state(cpu(iso),0.49,0.51)[1]
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
        label="Stationary distribution",
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


    savefig("/home/numerik/jkresse/code/CGEM/figures/ad_long_unbiased.png")

    sim = OpenMMSimulation()
    newdata = SimulationData(sim, xs, 1)
    data = ISOKANN.mergedata(iso.data, newdata)
    iso.data = data 
    iso =gpu(iso)
    run!(iso,10000)
end

ISOKANN.save(path_data*"iso_ad_long_enhanced.jld2", iso)
@save path_data*"ad_all_paths_long.jld2" xss
xss = load(path_data*"ad_all_paths_long.jld2")["xss"]

ISOKANN.savecoords(path_data*"/minimum_path_ad_long.pdb",iso,xs[:,1:100])

xs = ISOKANN.load_trajectory(path_data*"/minimum_path_ad_long.pdb")

using Plots
using Colors

# ---- color scheme  ----
c_iter1  = colorant"#D55E00"   # vermillion (distinct from orange)
c_iter5  = colorant"#E69F00"   # orange
c_iter10 = colorant"#009E73"   # teal
c_final  = :blue       

iters = [1, 5, 10, 15]
cols  = [c_iter1, c_iter5, c_iter10, c_final]

# Optional styling to make progression visible without reading legend
ms    = [3, 3, 4, 5]           # marker sizes
ma    = [0.55, 0.70, 0.85, 1.0]# marker alpha
lw    = [1.5, 2.0, 2.5, 3.0]   # line width (if you also want lines)
ls    = [:dash, :solid, :solid, :solid]

# ---- base scatter: sampling (traj) ----
p = scatter(
    ISOKANN.phi(traj), ISOKANN.psi(traj);
    markercolor = :grey,
    label = "Unbiased MD samples",   # (or "Unbiased MD training samples")
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
savefig("/home/numerik/jkresse/code/CGEM/figures/ad_long_iters.png")

using Statistics
using Plots

# wrap angular differences to [-pi, pi]
wrap_delta(x) = mod(x + pi, 2pi) - pi


rms_shift, max_jump = amore_convergence_phi_psi(xss; B=100)
iters = 2:length(rms_shift)+1

using Measures
p = plot(
    iters, rms_shift;
    marker=:circle,
    linewidth=3,
    label="RMS path shift",
    xlabel="AMORE-MD iteration",
    ylabel="Displacement in (\\phi, \\psi) [rad]",
    legendfontsize=26,
    tickfontsize=18,
    xlabelfontsize=26,
    ylabelfontsize=26,
    framestyle=:box,
    size = (1200, 1200),
    dpi = 300,
)

plot!(
    iters, max_jump;
    marker=:diamond,
    linewidth=3,
    linestyle=:dash,
    label="Maximum point displacement",
)


plot!(legend=:topright)
savefig("/home/numerik/jkresse/code/CGEM/figures/ad_long_convergence.png")


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

