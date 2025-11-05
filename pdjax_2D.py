# Assuming results = _solve(...) has been run
disp_x = results[0]  # Shape: (num_nodes,) - x-displacements
disp_y = results[1]  # Shape: (num_nodes,) - y-displacements
positions = params.pd_nodes  # Shape: (num_nodes, 2) - node positions

# Stack into full displacement vectors
disp = jnp.stack([disp_x, disp_y], axis=-1)  # Shape: (num_nodes, 2)
jax.debug.print("disp shape: {d}", d=disp.shape)

# Compute displacement magnitude (scalar for each node)
disp_magnitude = jnp.linalg.norm(disp, axis=1)  # Shape: (num_nodes,)

# Extract initial x/y positions for plotting
pos_x_init = positions[:, 0]
pos_y_init = positions[:, 1]

# Compute final (deformed) x/y positions
pos_x_final = positions[:, 0] + disp[:, 0]
pos_y_final = positions[:, 1] + disp[:, 1]

# Create the plot
fig, ax = plt.subplots()

# Plot initial positions as small black dots (time zero)
ax.scatter(pos_x_init, pos_y_init, c='orange', s=5, alpha=0.7, label='Initial Positions (t=0)')

# Plot final positions with displacement magnitude coloring
scatter = ax.scatter(pos_x_final, pos_y_final, c=disp_magnitude, cmap='viridis', s=10, alpha=0.8, label='Final Positions (End of Simulation)')

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Initial and Final Node Positions with Displacement Magnitude")
ax.legend()  # Add legend to distinguish initial and final
plt.colorbar(scatter, label='Displacement Magnitude')  # Colorbar for final positions
plt.tight_layout()
plt.show()


    ##################################################
# # Now using Optax to maximize
# random array of thickness values for initial thickness 
#key = jax.random.PRNGKey(0) 
#shape = (params.num_nodes,)  # Example shape
#param = jax.random.uniform(key, shape=shape, minval=0.5, maxval=1.5)

# scalar param and starting array of all 1's, optimizing thickness at every node
#param = jnp.array([1.0])
param = jnp.full((num_elems,), 1.0)

# optimizing only half of bar, such that thickness is symmetric
num_nodes = params.num_nodes
mid_index = num_nodes // 2  # midpoint of bar

no_damage_region_left = params.no_damage_region_left
no_damage_region_right = params.no_damage_region_right

middle_region = jnp.arange(jnp.max(no_damage_region_left) + 1, jnp.min(no_damage_region_right))
print("middle region: ", middle_region)

# region to optimize: left side *outside* no_damage_region
optimizable_indices = jnp.arange(0, mid_index)
optimizable_indices = optimizable_indices[~jnp.isin(optimizable_indices, no_damage_region_left)]

print("no_damage_region_left: ", no_damage_region_left)
print("no_damage_region_right: ", no_damage_region_right)
print("optimizable_indices: ", optimizable_indices)

param = jnp.ones((optimizable_indices.size,)) 
print("initial param: ", param)

loss_to_plot = []
damage_to_plot = []
strain_energy_to_plot =[]
damage = []

# when use strain energy density as loss, use smaller
#learning_rate = 1E-1

learning_rate = 1.0
num_steps = 20
thickness_min = 1.0E-2
thickness_max = 1.0E2

# Define gradient bounds
lower = 1E-2
upper = 20

#max_time = 1.0E-02
max_time = 5.0E-03
#max_time = 1.0E-03

# Optax optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(param)

# Optimization loop
damage_threshold = 0.5

# Loss function (already defined as 'loss')
#loss_and_grad = jax.value_and_grad(loss, argnums=2)
loss_and_grad = jax.value_and_grad(loss, argnums=2, has_aux=True)

# Clamp function
def clamp_params(grads):
	lower = 1E-05
	upper = 1.0E10
	#jax.debug.print("entering clamp_params: {t}", t=grads)
	grads = jax.tree_util.tree_map(lambda x: jnp.clip(jnp.abs(x), lower, upper), grads)
	#jax.debug.print("grad after clamping: {t}", t=grads)
	return grads

def make_symmetric_thickness(left_params):
    """Return full symmetric thickness array of shape (num_nodes,)."""
    left_fixed_thickness = 3.0
    right_fixed_thickness = 3.0

    # Mirror optimized section
    mirrored = left_params[::-1]
    middle_full = jnp.concatenate([left_params, mirrored])

    # Construct full bar thickness
    full_thickness = jnp.ones((num_nodes,))  # shape (num_nodes,)

    # Insert middle region
    full_thickness = full_thickness.at[middle_region].set(middle_full)

    # Fix the outer ends
    full_thickness = full_thickness.at[no_damage_region_left].set(left_fixed_thickness)
    full_thickness = full_thickness.at[no_damage_region_right].set(right_fixed_thickness)

    return full_thickness

# Optimization loop
for step in range(num_steps):
	def true_fn(thickness):
		jax.debug.print("thickness is all finite.")
		return thickness

	def false_fn(thickness):
		jax.debug.print("Non-finite thickness detected: {t}", t=thickness)
		return thickness

	full_thickness = make_symmetric_thickness(param)
	assert jnp.all(jnp.isfinite(param)), "Initial thickness contains NaNs!"

	# enforce fixed region if needed
	full_thickness = full_thickness.at[no_damage_region_left].set(1.0)
	full_thickness = full_thickness.at[no_damage_region_right].set(1.0)

	# Compute loss and gradients (grads wrt half param)
	(loss_val, (strain_energy, damage)), grads_full = loss_and_grad(
		params, state, full_thickness, 
		forces_array=forces_array, allow_damage=allow_damage, max_time=max_time
	)

	# Extract grads only for half region
	grads = grads_full[optimizable_indices]

	updates, opt_state = optimizer.update(grads, opt_state, param)
	param = optax.apply_updates(param, updates)
	param = jnp.clip(jnp.abs(param), 0.3, None)

	loss_to_plot.append(loss_val)
	strain_energy_to_plot.append(strain_energy)
	damage_to_plot.append(damage)

	jax.debug.print("Step {s}, loss={l}, thickness={t}",
					s=step, l=loss_val, t=full_thickness)
	print("damage in optimization loop: ", damage[-1])


##################################################
# # Now using Optax to maximize
# scalar param
#param = jnp.array([1.0])
thickness = jnp.full((params.num_nodes,), thickness0)
param = jnp.full((params.num_nodes,), density_field)

print("intitial param.shape: ", param.shape)

# optimizing only half of bar, such that thickness is symmetric
num_nodes = params.num_nodes
mid_index = num_nodes // 2  # midpoint of bar

no_damage_region_left = params.no_damage_region_left
no_damage_region_right = params.no_damage_region_right

# Get unique x-values (already sorted)
unique_x = jnp.unique(params.pd_nodes[:, 0])

# Define excluded x-values (leftmost and rightmost)
leftmost_x = unique_x[:5]   # First 5 x-values (left no-damage)
rightmost_x = unique_x[-5:] # Last 5 x-values (right no-damage)
excluded_x = jnp.concatenate([leftmost_x, rightmost_x])

# Define middle_region as nodes NOT in excluded x-values
middle_mask = ~jnp.isin(params.pd_nodes[:, 0], excluded_x)
middle_region = jnp.where(middle_mask)[0]

print("middle_region: ", middle_region)
print("middle_region size: ", middle_region.size)  # Should be ~300 for your gri

# Sort middle_region by x-position for symmetry
middle_region = middle_region[jnp.argsort(params.pd_nodes[middle_region, 0])]

# Define optimizable_indices as the left half of middle_region
mid_middle = len(middle_region) // 2
optimizable_indices = middle_region[:mid_middle]

# Initialize param with the correct size
param = jnp.ones((mid_middle,))

loss_to_plot = []
damage_to_plot = []
strain_energy_to_plot = []

learning_rate = 1.0
num_steps = 3
density_min = 0.0
density_max = 1.0

# Define gradient bounds
lower = 1E-2
upper = 20

max_time = 1.0E-04

# Optax optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(param)

# Optimization loop
damage_threshold = 0.5

# Loss function (already defined as 'loss')
loss_and_grad = jax.value_and_grad(loss, argnums=2)

# Clamp function
def clamp_params(grads):
	lower = 1E-05
	upper = 1.0E10
	#jax.debug.print("entering clamp_params: {t}", t=grads)
	grads = jax.tree_util.tree_map(lambda x: jnp.clip(jnp.abs(x), lower, upper), grads)
	#jax.debug.print("grad after clamping: {t}", t=grads)
	return grads

def make_symmetric_density(left_params):
    """Return full symmetric density array of shape (num_nodes,)."""
    left_fixed_density = 1.0
    right_fixed_density = 1.0

    # Mirror optimized section
    mirrored = left_params[::-1]
    middle_full = jnp.concatenate([left_params, mirrored])

    # Construct full bar density field
    full_density_field = jnp.ones((num_nodes,))  # shape (num_nodes,)

    # Insert middle region
    full_density_field = full_density_field.at[middle_region].set(middle_full)

    # Fix the outer ends
    full_density_field = full_density_field.at[no_damage_region_left].set(left_fixed_density)
    full_density_field = full_density_field.at[no_damage_region_right].set(right_fixed_density)

    return full_density_field

# Optimization loop
for step in range(num_steps):
	def true_fn(thickness):
		jax.debug.print("thickness is all finite.")
		return thickness

	def false_fn(thickness):
		jax.debug.print("Non-finite thickness detected: {t}", t=thickness)
		return thickness

	full_density_field = make_symmetric_density(param)
	assert jnp.all(jnp.isfinite(param)), "Initial density contains NaNs!"

	# enforce fixed region if needed
	full_density_field = full_density_field.at[no_damage_region_left].set(1.0)
	full_density_field = full_density_field.at[no_damage_region_right].set(1.0)

	# Compute loss and gradients (grads wrt half param)
	loss_val, grads_full = loss_and_grad(
        params, state, thickness, full_density_field, 
        forces_array=forces_array, allow_damage=allow_damage, max_time=max_time)

	# Extract grads only for half region
	grads = grads_full[optimizable_indices]

	updates, opt_state = optimizer.update(grads, opt_state, param)
	param = optax.apply_updates(param, updates)
	param = jnp.clip(jnp.abs(param), 0.3, None)
 
    # Now compute strain_energy and damage separately for plotting
	output_vals = _solve(params, state, thickness, full_density_field, forces_array, allow_damage, max_time)
	strain_energy = output_vals.strain_energy  # Adjust index if needed (check PDState)
	damage = output_vals.damage  # Adjust index if needed
    
	loss_to_plot.append(loss_val)
	strain_energy_to_plot.append(strain_energy)
	damage_to_plot.append(damage)
    
	jax.debug.print("Step {s}, loss={l}, density_field.sum={t}", s=step, l=loss_val, t=full_density_field.sum())
	print("damage in optimization loop: ", damage[-1])
