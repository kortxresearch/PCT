from getdist import MCSamples
import numpy as np

samples = MCSamples(root='chains/planck2018_running')
alpha_s = samples.getParams().alpha_s

print(f"α_s mean:   {np.mean(alpha_s):.4f}")
print(f"α_s median: {np.median(alpha_s):.4f}")
print(f"68% CI:     [{np.percentile(alpha_s, 16):.4f}, {np.percentile(alpha_s, 84):.4f}]")
print(f"95% CI:     [{np.percentile(alpha_s, 2.5):.4f}, {np.percentile(alpha_s, 97.5):.4f}]")