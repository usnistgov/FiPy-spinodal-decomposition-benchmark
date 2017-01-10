# coding: utf-8

# # 1b. Fixed-flux spinodal decomposition on a square domain

# The chemical free energy is given by
# 
# $$ f_{chem}\left[ c \left( \vec{r} \right) \right] = f_0\left[ c \left( \vec{r} \right) \right] = {\rho}_s (c - c_{\alpha})^2 (c_{\beta} - c)^2 $$
# 
# In FiPy we write the evolution equation as 
# 
# $$ \frac{\partial c}{\partial t} = \nabla \cdot  \left[
#        D \left( c \right) \left( \frac{ \partial^2 f_0 }{ \partial c^2} \nabla c - \kappa \nabla \nabla^2 c \right)   \right] $$

from __future__ import print_function

import fipy as fp
import glob
import json
import numpy as np
import os
import sys
from fipy.solvers.pysparse import LinearLUSolver as Solver


problem = '1'
domain = 'b'
nx = 200
dx = 1.0

# The first step in implementing any problem in FiPy is to define the mesh. For [Problem 1b]({{ site.baseurl }}/hackathon1/#a.-Square-Periodic) the solution domain is just a square domain with fixed boundary conditions, so a `Grid2D` object is used. No other boundary conditions are required.

mesh = fp.Grid2D(nx=nx, ny=nx, dx=dx, dy=dx)

# The next step is to define the parameters and create a solution variable.
# Constants and initial conditions:
# $c_{\alpha}$ and $c_{\beta}$ are concentrations at which the bulk free energy has minima.
# $\kappa$ is the gradient energy coefficient.
# $\varrho_s$ controls the height of the double-well barrier.

c_alpha = 0.3
c_beta = 0.7
kappa = 2.0
M = 5.0
c_0 = 0.5
epsilon = 0.01
rho_s = 5.0

c_var = fp.CellVariable(mesh=mesh, name=r"$c$", hasOld=True)

c_var.mesh.cellCenters()

# Now we need to define the initial conditions.
# 
# $c_{var}$ is a cell variable specifying concentrations at various points in the mesh. Here, it is the solution variable:
# 
# Set $c\left(x, y\right)$ such that
# 
# $$ c\left(x, y\right) = \bar{c}_0 + \epsilon_{\Box}[\cos(0.105x) \cos(0.11y) + [\cos(0.13x) \cos(.087y)]^2 + \cos(0.025x-0.15y) \cos(0.07x-0.02y)].$$

# array of sample c-values: used in f versus c plot
vals = np.linspace(-.1, 1.1, 1000)

c_var = fp.CellVariable(mesh=mesh, name=r"$c$", hasOld=True)

x , y = np.array(mesh.x), np.array(mesh.y)

c_var[:] = c_0 + epsilon * (np.cos(0.105 * x) * np.cos(0.11 * y) + (np.cos(0.13 * x) * np.cos(0.087 * y))**2 + np.cos(0.025 * x - 0.15 * y) * np.cos(0.07 * x - 0.02 * y))

# bulk free energy density
def f_0(c):
    return rho_s * (c - c_alpha)**2 * (c_beta-c)**2
def f_0_var(c_var):
    return 2 * rho_s * ((c_alpha - c_var)**2 + 4*(c_alpha - c_var)*(c_beta - c_var) + (c_beta - c_var)**2)
# free energy
def f(c):
    return f_0(c)+ .5*kappa*(c.grad.mag)**2


# Here, the elapsed time, total free energy, and concentration cell variable are saved to separate lists at designated time intervals. These lists are then updated in an .npz file in a directory of your choice.

# save elapsed time and free energy at each data point
time_data = []
cvar_data = []
f_data = []

def save_data(time, cvar, f, step):
    time_data.append(time)
    cvar_data.append(np.array(cvar.value))
    f_data.append(f.value)
    
    file_name = "data/1{0}{1}_{2}".format(domain, nx, str(step).rjust(5, '0'))
    np.savez(file_name, time = time_data, c_var = cvar_data, f = f_data)


# ## Define the Equation

eqn = fp.TransientTerm(coeff=1.) == fp.DiffusionTerm(M * f_0_var(c_var)) - fp.DiffusionTerm((M, kappa))


# ## Solve the Equation

# To solve the equation a simple time stepping scheme is used which is decreased or increased based on whether the residual decreases or increases. A time step is recalculated if the required tolerance is not reached. In addition, the time step is kept under 1 unit. The data is saved out every 10 steps.

elapsed = 0.0
steps = 0
dt = 0.01
dt_max = 1.0
total_sweeps = 2
tolerance = 1e-1
total_steps = int(sys.argv[1])
checkpoint = int(sys.argv[2])
duration = 900.0


c_var.updateOld()
solver = Solver()

while elapsed < duration and steps < total_steps:
    res0 = eqn.sweep(c_var, dt=dt, solver=solver)

    for sweeps in range(total_sweeps):
        res = eqn.sweep(c_var, dt=dt, solver=solver)

    if res < res0 * tolerance:
        # anything in this loop will only be executed every $checkpoint steps
        if (steps % checkpoint == 0):
            save_data(elapsed, c_var, f(c_var).cellVolumeAverage*mesh.numberOfCells*dx*dx, steps)
            
        steps += 1
        elapsed += dt
        dt *= 1.1
        dt = min(dt, dt_max)
        c_var.updateOld()
    else:
        dt *= 0.8
        c_var[:] = c_var.old
    

# ## Free Energy Plots

newest = max(glob.iglob('data/1{0}{1}*.npz'.format(domain, nx)), key=os.path.getctime)

times = np.load(newest)['time']
f = np.load(newest)['f']
data = [dict(time=time, free_energy=energy) for time, energy in zip(times, f)]

# Dump json to string
print("data:")
print("  # Gather simulation output")
print("  - name: free_energy")
print("    # JSON list of {time, energy} pairs")
print("    values: ", end="")
print(json.dumps(data))

## Dump json to disk
#with open('myjson.json', 'w') as fp:
#    json.dump(data, fp, indent=2)
#print("data:")
#print("  # Gather simulation output")
#print("  - name: free_energy")
#print("    # JSON list of {time, energy} pairs")
#print("    url: myjson.json")
#print("    type: json")
