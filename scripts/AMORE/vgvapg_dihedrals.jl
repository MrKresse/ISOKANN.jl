using Revise
using ISOKANN
using MultivariateStats
using Clustering
using Distances
using Statistics
using Optim
using PyCall
using BioStructures


iso = ISOKANN.load("/scratch/htc/jkresse/transfer/iso_vgvapg_1us.jld2")

coords = iso.data.coords[1]
sim = iso.data.sim

featurizer = OpenMM.FeaturesAngles(sim)

data = SimulationData(sim,coords,1;featurizer=featurizer)

iso = Iso(data, opt=NesterovRegularized(), minibatch=1000)
run!(iso,20_000)
ISOKANN.plot_training(iso)
ISOKANN.save("/data/numerik/people/jkresse/CGEM/iso_vgvapg_angles.jld2",iso)

iso=cpu(iso)

intpairs = Vector{Tuple{Int, Int}}()
for i in 1:73
  for j in i+1:73
      push!(intpairs, (i, j))
  end
end
distances = OpenMM.FeaturesPairs(intpairs)

features_distance = distances(coords)
endtoend = features_distance[72,:]
chi_vals = cpu(ISOKANN.chis(iso))
chi_vals = [chi_val[1] for chi_val in eachcol(chi_vals) ]

scatter(chi_vals,endtoend)
scatter(iso.data.features[1][2,:], iso.data.features[1][8,:]; xlim  = (-pi,pi), ylim = (-pi,pi), size= (1200,1200), marker_z= chi_vals)

grad = featureGradient(iso,iso.data.coords[1])
avggrad = mean(grad.^2,dims=2)
scatter(avggrad)
savefig("/home/numerik/jkresse/code/CGEM/figures/vgvapg_grad_angles")
# Step 1: Compute mean squared value per feature
mean_squares = mean(avggrad, dims=2)
# Convert to 1D vector for convenience
ms = vec(mean_squares)

# Step 2: Visual inspection of distribution
histogram(ms, bins=100,  xlabel="Mean square", ylabel="Frequency",
          title="Distribution of Feature Mean Squares", yscale=:log10)
