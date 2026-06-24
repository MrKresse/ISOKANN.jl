using Revise
using ISOKANN
using MultivariateStats
using Clustering
using Distances
using Statistics
using JLD2
using Optim

intpairs = Vector{Tuple{Int, Int}}()

for i in 1:73
  for j in i+1:73
      push!(intpairs, (i, j))
  end
end
featurizer = OpenMM.FeaturesPairs(intpairs)

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<MD>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
sim = OpenMMSimulation(pdb="/home/numerik/jkresse/code/bachelor/vgvapg_unfolded_processed.pdb", steps =250,forcefields=OpenMM.FORCE_AMBER_IMPLICIT)
traj = OpenMM.trajectory(sim,500_000_000;saveevery=10_000)
@save "/scratch/htc/jkresse/transfer/traj_vgvapg_implicit_1us.jld2" traj

data = SimulationData(sim, traj,1; featurizer=featurizer)

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ISOKANN>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
iso = Iso(data, opt=NesterovRegularized(), minibatch=1000)
iso = gpu(iso)
run!(iso,20_000)
ISOKANN.plot_training(iso)

iso = ISOKANN.load("/scratch/htc/jkresse/transfer/iso_vgvapg_1us.jld2")

#<<<<<<<<<<<<enhanced sampling>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
iso=gpu(iso)
xss=nothing

for _ in 1:100
    xs_samples = iso.data.coords[1]
    chi_vals = cpu(ISOKANN.chis(iso))
    chi_vals = [chi_val[1] for chi_val in eachcol(chi_vals) ]
    x0s = xs_samples[:,inds]
    j = rand(1:size(x0s, 2))

    xs = reactionpath_minimum(iso,x0s[:,j];steps=200,f_reltol=1e-5, alphaguess =5e-4, iterations=350,show_trace=false, algorithm = Optim.ConjugateGradient )
    xss = isnothing(xss) ? xs : hcat(xss,xs)


    data = ISOKANN.addcoords(iso.data,xs)
    iso.data=data
    run!(iso,2000)


    phi_sample = [ISOKANN.phi(x, [24, 26, 28, 40]) for x in eachcol(xs_samples[:,1:10_000])]
    psi_sample = [ISOKANN.psi(x, [26, 28, 40, 42])  for x in eachcol(xs_samples[:,1:10_000])]

    phi_path = [ISOKANN.phi(x, [24, 26, 28, 40]) for x in eachcol(xs)]
    psi_path = [ISOKANN.psi(x, [26, 28, 40, 42])  for x in eachcol(xs)]

    gr()
    p = scatter(phi_sample, psi_sample,
        #markerz=[iso.model(iso.data.featurizer(x))[1] for x in eachcol(traj)],
        markercolor=:grey,
        label="Stationary distribution",
        framestyle=:box,
        size=(1200, 1200),     # Figure size in pixels
        dpi=300,             # Higher resolution
        markersize=3,        # Marker size in points
        markerstrokewidth=0,
        markeralpha=1,
        xlabel="\\phi",
        ylabel="\\psi",
        title="", xlabelfontsize=26,
        ylabelfontsize=26,
        legendfontsize=26,
        tickfontsize=18, xlim = (-pi,pi), ylim=(-pi,pi))

    scatter!(phi_path,psi_path;
        markercolor=:blue,
        markersize=6,
        label="MEP states",
        title="",
        markerstrokewidth=0)
    display(p)
end
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<converged pathways>>>>>>>>>>>>>>>>>>>>>
xs_samples = iso.data.coords[1]
chi_vals = cpu(ISOKANN.chis(iso))
chi_vals = [chi_val[1] for chi_val in eachcol(chi_vals) ]

inds = (chi_vals .> 0.4995) .& (chi_vals .< 0.5005)
x0s = xs_samples[:,inds]
xss = nothing
iso=cpu(iso)
for x in eachcol(x0s)
    xs = reactionpath_minimum(iso,x;steps=200,f_reltol=1e-5, alphaguess =5e-4, iterations=350,show_trace=false, algorithm = Optim.ConjugateGradient )
    xss = isnothing(xss) ? xs : hcat(xss,xs)
end
@save "/data/numerik/people/jkresse/CGEM/vgvapg_transition_enhanced.jld2" xss 
xss = load("/data/numerik/people/jkresse/CGEM/vgvapg_transition_enhanced.jld2")["xss"]

ISOKANN.save("/data/numerik/people/jkresse/CGEM/iso_vgvapg_enhanced.jld2",iso)
iso = ISOKANN.load("/data/numerik/people/jkresse/CGEM/iso_vgvapg_enhanced.jld2")

phi_sample = [ISOKANN.phi(x, [24, 26, 28, 40]) for x in eachcol(xs_samples[:,1:10000])]
psi_sample = [ISOKANN.psi(x, [26, 28, 40, 42])  for x in eachcol(xs_samples[:,1:10000])]

phi_path = [ISOKANN.phi(x, [24, 26, 28, 40]) for x in eachcol(xss)]
psi_path = [ISOKANN.psi(x, [26, 28, 40, 42])  for x in eachcol(xss)]

phi = [ISOKANN.phi(x, [24, 26, 28, 40]) for x in eachcol(x0s)]
psi = [ISOKANN.psi(x, [26, 28, 40, 42])  for x in eachcol(x0s)]

ticks = [-pi, -3*pi/4, -pi/2, -pi/4, 0, pi/4, pi/2, 3*pi/4, pi]
gr()


using Plots

vals   = Float64.([-π, -π/2, 0, π/2, π])
labels = ["-π", "-π/2", "0", "π/2", "π"]
tickspec = (vals, labels)

p = scatter(phi_sample, psi_sample;
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
    yticks=tickspec,
)


scatter!(phi_path[1215:1216],psi_path[1215:1216];
    markercolor=:blue,
    markersize=6,
    label="\\chi - MEP states",
    markerstrokewidth=0)
display(p)
scatter!(phi,psi;
    marker=:diamond,
    markercolor=:magenta,
    markersize=10,
    label="Initial states",
    title="",
    markeralpha=1,markerstrokewidth=0)

savefig("/home/numerik/jkresse/code/CGEM/figures/fig2_ramachandran")

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Kinetic bottleneck>>>>>>>>>>>>>>>>>>>>>>>
iso = cpu(iso)
avggrad = averageGradient(iso;xs =iso.data.coords[1][:,1:10000])
savefig("/home/numerik/jkresse/code/CGEM/figures/vgvapg_grad.png")
F, chiv = marginal_free_energy(iso;coords=iso.data.coords[1][:,1:10000])
plot(chiv,F)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<local vs global>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
mses= zeros(13)
num_atoms=73
num_frames=200
i=200
for i in 200:200:2600
    xs = xss[:,i-199:i]
    localgrad=coordGradient(iso,xs)
    j= Int(i/200)
    
    reshaped = reshape(localgrad, (3, num_atoms, num_frames))
    grad_squared = sum(reshaped .^ 2, dims=1)[1, :, :]
    gradients = grad_squared
    @save "/data/numerik/people/jkresse/CGEM/vgvapg_gradient$j.jld2" gradients
    mse = mean((avggrad .- grad_squared[:,1:2:end]).^2)
    mses[j] =mse
end


iso.data.coords[1]
iso.data.coords[2]
reactive_path
