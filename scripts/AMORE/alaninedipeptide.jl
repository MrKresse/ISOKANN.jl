using Revise
using ISOKANN
using MultivariateStats
using Clustering
using Distances
using Optim
using JLD2

sim = OpenMMSimulation()

intpairs = Vector{Tuple{Int, Int}}()

for i in 1:22
  for j in i+1:22
      push!(intpairs, (i, j))
  end
end
featurizer = OpenMM.FeaturesPairs(intpairs)


scatter_ramachandran(traj)
data = trajectorydata_bursts(sim,10000,1)
iso = Iso(data,opt=NesterovRegularized(),minibatch = 1000)
iso =gpu(iso)
run!(iso,4000)

ISOKANN.plot_training(iso)
scatter_ramachandran(iso)
ISOKANN.save(path_data*"iso_ad.jld2", iso)
iso = ISOKANN.load(path_data*"iso_ad.jld2")
#################Extracting the minimum path################################
traj = iso.data.coords[1]
transition= transition_state(cpu(iso),0.49,0.51)[1]#34 states
@time transition_min =[ISOKANN.energyminimization_chilevel(cpu(iso),x;f_reltol=1e-5, alphaguess =1e-5, iterations=250,show_trace=true, algorithm = Optim.ConjugateGradient) for x in eachcol(transition)]
transition_min = reduce(hcat,transition_min)


xss=nothing
for i in 1:size(transition_min,2)
    xs = reactionpath_minimum(iso, transition_min[:,i];steps=100,f_reltol=1e-5, alphaguess =1e-5, iterations=250,show_trace=false, algorithm = Optim.ConjugateGradient)
    xss = isnothing(xss) ? xs : hcat(xss,xs)
end  
@save path_data*"ad_all_paths.jld2" xss
xss = load(path_data*"ad_all_paths.jld2")["xss"]

phi = [ISOKANN.phi(x) for x in eachcol(transition)]
psi = [ISOKANN.psi(x) for x in eachcol(transition)]

phi_path = [ISOKANN.phi(x) for x in eachcol(xss)]
psi_path = [ISOKANN.psi(x)  for x in eachcol(xss)]

vals   = Float64.([-π, -π/2, 0, π/2, π])
labels = ["-π", "-π/2", "0", "π/2", "π"]
tickspec = (vals, labels)


p = scatter(ISOKANN.phi(traj), ISOKANN.psi(traj),
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


savefig("/home/numerik/jkresse/code/CGEM/figures/f1_ramachandran.png")
scatter_ramachandran!(xs[:,9:10])
scatter_ramachandran!(xs[:,49:50])
scatter_ramachandran!(xs[:,94:95])
savefig("/home/numerik/jkresse/code/CGEM/figures/f1_ramachandran_states.png")

##############free energy##############################################

plot_free_energy_with_fixman(iso;coords=traj)
savefig("/home/numerik/jkresse/code/CGEM/figures/f1_free_energy.png")
##########################################################################

ISOKANN.savecoords(path_data*"/minimum_path_ad.pdb",iso,xs[:,1:100])

xs = ISOKANN.load_trajectory(path_data*"/minimum_path_ad.pdb")
#xx = hcat(x,x)#workaround to be able to save just one frame
#ISOKANN.savecoords(path_data*"/start_point_dialanine.pdb",iso,xx)

################Visualizing the gradient###################################
iso = cpu(iso)

#gradients = weightedHeatmap(iso,xs)



gradc= averageGradient(iso;xs=iso.data.coords[1],nbins=100)

gradients=gradc
@save path_data*"/norm_gradient_ad.jld2" gradients
savefig("/home/numerik/jkresse/code/CGEM/figures/heatmap_ad.png")

rankAtomImportance(gradc)

gradc= coordGradient(iso,xs)

sumHeatmap(gradc)
gradf= featureGradient(iso,xs)
featureHeatmap(gradf)
saveSum(gradc;out=path_data*"/norm_gradient_path.jld2")

intpairs[argmin(sum(gradf,dims=2))]
intpairs[argmax(sum(gradf,dims=2))]
#start pymol:
#run /home/numerik/jkresse/code/bachelor/pymolgradient.py
#path_pdb = "/data/numerik/people/jkresse/CGEM/minimum_path_ad.pdb"
#path_gradient = "/data/numerik/people/jkresse/CGEM/norm_gradient_ad.jld2"
#showFramewiseGradient(path_gradient, path_pdb)