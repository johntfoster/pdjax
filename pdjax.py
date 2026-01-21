from functools import partial
import time
import math

import jax
import jax.numpy as jnp

import numpy as np
import scipy.spatial
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.animation as animation



import jax.scipy
import jax.scipy.optimize
from jax.scipy.optimize import minimize
from jax.nn import softplus

from jax import grad, jit


from typing import Union, Tuple

import optax
from jax import jit, vmap, grad, value_and_grad
from jax import lax
from jax import grad
from jax.experimental import checkify


import jax
import jax.numpy as jnp
import numpy as np
import scipy.spatial
from typing import Union, NamedTuple, Optional


# ----------------------------
# PARAMETER STRUCT
# ----------------------------
class PDParams(NamedTuple):
	bar_length: float
	number_of_elements: int
	bulk_modulus: float
	elastic_modulus: float
	density: float
	thickness: jnp.ndarray
	horizon: float
	critical_stretch: Optional[float]
	prescribed_velocity: Optional[float]
	prescribed_force: Optional[float]
	nodes: jnp.ndarray
	lengths: jnp.ndarray
	pd_nodes: jnp.ndarray
	num_nodes: int
	neighborhood: jnp.ndarray
	reference_position_state: jnp.ndarray
	reference_magnitude_state: jnp.ndarray
	num_neighbors: jnp.ndarray
	max_neighbors: int
	no_damage_region_left: jnp.ndarray
	no_damage_region_right: jnp.ndarray
	width: float  
	right_bc_region: jnp.ndarray
	left_bc_region: jnp.ndarray
	undamaged_influence_state_left: jnp.ndarray
	undamaged_influence_state_right: jnp.ndarray


class PDState(NamedTuple):
	disp: jnp.ndarray
	vel: jnp.ndarray
	acc: jnp.ndarray
	vol_state: jnp.ndarray
	rev_vol_state: jnp.ndarray
	influence_state: jnp.ndarray
	undamaged_influence_state: jnp.ndarray
	damage: jnp.ndarray
	forces_array: jnp.ndarray
	disp_array: jnp.ndarray
	velo_array: jnp.ndarray
	strain_energy: float
	time: float


# ----------------------------
# GLOBAL INITIALIZATION FUNCTION
# ----------------------------
def init_problem(bar_length: float = 20.0,
				 density: float = 1.0,
				 bulk_modulus: float = 100.0,
				 elastic_modulus: float = 300.0,
				 number_of_elements: int = 20,
				 horizon: Optional[float] = None,
				 thickness: Union[float, np.ndarray] = 1.0,
				 prescribed_velocity: Optional[float] = None,
				 prescribed_force: Optional[float] = None,
				 critical_stretch: Optional[float]= None):

	
	"""
	Create PDParams and PDState tuples for a new problem.
	"""

	delta_x = bar_length / number_of_elements
	if horizon is None:
		horizon = delta_x * 3.015

	# thickness as array
	if isinstance(thickness, float) or np.isscalar(thickness):
		thickness_arr = jnp.ones(number_of_elements) * thickness
	elif isinstance(thickness, (np.ndarray, jnp.ndarray)):
		thickness_arr = jnp.asarray(thickness)
		assert thickness_arr.shape[0] == number_of_elements, ValueError("Thickness array length must match number of elements")
	else:
		raise ValueError("thickness must be a float or array")
	
	# nodes and element lengths
	nodes = jnp.linspace(-bar_length / 2.0, bar_length / 2.0, num=number_of_elements + 1)
	lengths = jnp.array(nodes[1:] - nodes[0:-1])
	pd_nodes = jnp.array(nodes[0:-1] + lengths / 2.0)
	num_nodes = pd_nodes.shape[0]

	# kdtree setup
	tree = scipy.spatial.cKDTree(pd_nodes[:, None])
	reference_magnitude_state, neighborhood = tree.query(
		pd_nodes[:, None], k=100, p=2, eps=0.0,
		distance_upper_bound=(horizon + np.max(lengths) / 2.0))

	# trim out self-distance column
	reference_magnitude_state = jnp.delete(reference_magnitude_state, 0, 1)

	num_neighbors = jnp.asarray((neighborhood != tree.n).sum(axis=1)) - 1
	max_neighbors = int(np.max((neighborhood != tree.n).sum(axis=1)))

	neighborhood = jnp.asarray(neighborhood[:, :max_neighbors])
	reference_magnitude_state = reference_magnitude_state[0:, :max_neighbors - 1]

	row_indices = jnp.arange(neighborhood.shape[0]).reshape(-1, 1)
	neighborhood = jnp.where(neighborhood == tree.n, row_indices, neighborhood)
	neighborhood = jnp.delete(neighborhood, 0, 1)

	reference_position_state = pd_nodes[neighborhood] - pd_nodes[:, None]
	reference_magnitude_state = jnp.where(reference_magnitude_state == np.inf, 0.0, reference_magnitude_state)
	#jax.debug.print("reference_magnitude_state = {r}", r=reference_magnitude_state)

	if prescribed_velocity is not None and prescribed_force is not None:
		raise ValueError("Only one of prescribed_velocity or prescribed_force should be set, not both.")
	
	if prescribed_velocity is None and prescribed_force is None:
		raise ValueError("Either prescribed_velocity or prescribed_force must be set.")

	#:The node indices of the boundary region at the left end of the bar
	li = 0
	left_bc_mask = neighborhood[li] != li
	left_bc_region = neighborhood[li][left_bc_mask]

	#:The node indices of the boundary region at the right end of the bar
	ri = num_nodes - 1
	right_bc_mask = neighborhood[ri] != ri
	right_bc_region = neighborhood[ri][right_bc_mask]

	#left_bc_region = jnp.asarray(tree.query_ball_point(pd_nodes[0, None], r=(2.0 * horizon ), p=2, eps=0.0)).sort()

	#right_bc_region = jnp.asarray(tree.query_ball_point(pd_nodes[-1, None], r=(2.0 * horizon ), p=2, eps=0.0)).sort()


	if prescribed_velocity is not None:
		left_bc_region = jnp.asarray(tree.query_ball_point(pd_nodes[0, None], r=(horizon + np.max(lengths) / 2.0), p=2, eps=0.0)).sort()
		right_bc_region = jnp.asarray(tree.query_ball_point(pd_nodes[-1, None], r=(horizon + np.max(lengths) / 2.0), p=2, eps=0.0)).sort()

	# no-damage regions
    # set to 4.0 for previous results, but for scalar bar trying 2.0
	no_damage_region_left = jnp.asarray(
		tree.query_ball_point(pd_nodes[0, None], r=(4.0 * horizon + np.max(lengths) / 2.0), p=2, eps=0.0)
	).sort()
	no_damage_region_right = jnp.asarray(
		tree.query_ball_point(pd_nodes[-1, None], r=(4.0 * horizon + np.max(lengths) / 2.0), p=2, eps=0.0)
	).sort()


	# initial vol_state (full volume)
	vol_state = jnp.ones((num_nodes, max_neighbors - 1))
	rev_vol_state = vol_state.copy()

	#jax.debug.print("vol_state in init: {v}",v=vol_state)

	influence_state = jnp.where(vol_state > 1.0e-16, 1.0, 0.0)
	undamaged_influence_state = influence_state.copy()
	
	undamaged_influence_state_left = influence_state.at[no_damage_region_left, :].get()
	undamaged_influence_state_right = influence_state.at[no_damage_region_right, :].get()

	width = 1.0  # Width of the bar, can be adjusted if needed


	# package params
	params = PDParams(
		bar_length, number_of_elements, bulk_modulus, elastic_modulus, density, thickness_arr,
		horizon, critical_stretch, prescribed_velocity, prescribed_force,
		nodes, lengths, pd_nodes, num_nodes, neighborhood,
		reference_position_state, reference_magnitude_state, num_neighbors, max_neighbors,
		no_damage_region_left, no_damage_region_right, width, right_bc_region, left_bc_region,
		undamaged_influence_state_left, undamaged_influence_state_right
	)


	# package initial state
	state = PDState(
		disp=jnp.zeros(num_nodes),
		vel=jnp.zeros(num_nodes),
		acc=jnp.zeros(num_nodes),
		vol_state=vol_state,
		rev_vol_state=rev_vol_state,
		influence_state=influence_state,
		undamaged_influence_state=undamaged_influence_state,
		forces_array=jnp.zeros(num_nodes),
		disp_array=jnp.zeros(num_nodes),
		velo_array=jnp.zeros(num_nodes),
		strain_energy=0.0,
		damage=jnp.zeros(num_nodes),
		time=0.0
	)

	return params, state


############### global functions to replace jnp.where functions in compute_partial_volumes ###############
@jax.jit
def vol_state_uncorrected_where(ref_mag_state: jax.Array, vol_state_uncorrected: jax.Array):
	"""
	Replace entries in vol_state_uncorrected with 0.0 where ref_mag_state < 1.0e-16,
	using lax.cond and nested vmap for full JIT/grad compatibility.
	Works for 1D or 2D arrays.
	"""
	# Scalar conditional
	def cond_fn(r, v):
		return jax.lax.cond(r < 1e-16, lambda _: 0.0, lambda _: v, operand=None)

	# Vectorize over columns (neighbors)
	col_fn = jax.vmap(cond_fn, in_axes=(0, 0))
	# Vectorize over rows (nodes)
	return jax.vmap(col_fn, in_axes=(0, 0))(ref_mag_state, vol_state_uncorrected)


@jax.jit
def vol_state_where(ref_mag_state: jax.Array, horiz: float, lens: jax.Array, neigh: jax.Array, vol_state_uncorrected: jax.Array):
	"""
	Replace entries in vol_state_uncorrected with 0.0 where
	ref_mag_state >= horiz + lens[neigh] / 2.0,
	using lax.cond and nested vmap.
	Works for 1D or 2D arrays.
	"""
	threshold = horiz + lens[neigh] / 2.0  # shape matches ref_mag_state

	# Scalar conditional
	def cond_fn(r, thresh, v):
		return jax.lax.cond(r < thresh, lambda _: v, lambda _: 0.0, operand=None)

	# Vectorize over columns (neighbors)
	col_fn = jax.vmap(cond_fn, in_axes=(0, 0, 0))
	# Vectorize over rows (nodes)
	return jax.vmap(col_fn, in_axes=(0, 0, 0))(ref_mag_state, threshold, vol_state_uncorrected)


def vol_state_clip_where(vol_state: jax.Array, vol_state_uncorrected: jax.Array):
    return jnp.minimum(vol_state, vol_state_uncorrected)

######################################################################################
def compute_partial_volumes(params, thickness:jax.Array):

	# Setup some local (to function) convenience variables
	neigh = params.neighborhood
	lens = params.lengths
	ref_mag_state = params.reference_magnitude_state
	horiz = params.horizon


	# Initialize the volume_state to the lengths * width * thickness
	width = params.width
	vol_state_uncorrected = lens[neigh] * thickness[neigh] * width 

	#Zero out entries that are not in the family
	#vol_state_uncorrected = jnp.where(ref_mag_state < 1.0e-16, 0.0, vol_state_uncorrected) 
	vol_state_uncorrected = vol_state_uncorrected_where(ref_mag_state, vol_state_uncorrected)


	#vol_state = jnp.where(ref_mag_state < horiz + lens[neigh] / 2.0, vol_state_uncorrected, 0.0)
	vol_state = vol_state_where(ref_mag_state, horiz, lens, neigh, vol_state_uncorrected)  

	# Check to see if the neighboring node has a partial volume
	is_partial_volume = jnp.abs(horiz - ref_mag_state) < lens[neigh] / 2.0
	#jax.debug.print("Any NaNs? {y}", y=jnp.any(jnp.isnan(is_partial_volume)))

	# Two different scenarios:
	is_partial_volume_case1 = is_partial_volume * (ref_mag_state >= horiz)
	is_partial_volume_case2 = is_partial_volume * (ref_mag_state < horiz)

	# Compute the partial volumes conditionally
	vol_state = jnp.where(is_partial_volume_case1, (lens[neigh] / 2.0 - (ref_mag_state - horiz)) * width * thickness[neigh], vol_state)
	vol_state = jnp.where(is_partial_volume_case2, (lens[neigh] / 2.0 + (horiz - ref_mag_state)) * width * thickness[neigh], vol_state)


	# If the partial volume is predicted to be larger than the unocrrected volume, set it back
	#vol_state = jnp.where(vol_state > vol_state_uncorrected, vol_state_uncorrected, vol_state)
	vol_state = vol_state_clip_where(vol_state, vol_state_uncorrected)
	EPS_vol = 1e-6
	vol_state = jnp.maximum(vol_state, EPS_vol)


	# Now compute the "reverse volume state", this is the partial volume of the "source" node, i.e. node i,
	# as seen from node j.  This doesn't show up anywhere in any papers, it's just used here for computational
	# convenience
	vol_array = lens[:,None] * width * thickness[:, None]
	#jax.debug.print("Any NaNs? {y}", y=jnp.any(jnp.isnan(vol_array)))

	rev_vol_state = jnp.ones_like(vol_state) * vol_array
	#jax.debug.print("Any NaNs? {y}", y=jnp.any(jnp.isnan(rev_vol_state)))
	
	rev_vol_state = jnp.where(is_partial_volume_case1, (lens[:, None] / 2.0 - (ref_mag_state - horiz)) * width * thickness[:, None], rev_vol_state)
	rev_vol_state = jnp.where(is_partial_volume_case2, (lens[:, None] / 2.0 + (horiz - ref_mag_state)) * width * thickness[:, None], rev_vol_state)



	#If the partial volume is predicted to be larger than the uncorrected volume, set it back
	#rev_vol_state = jnp.where(rev_vol_state > vol_array, vol_array, rev_vol_state)
	rev_vol_state = vol_state_clip_where(rev_vol_state, vol_array)
	rev_vol_state = jnp.maximum(rev_vol_state, EPS_vol)

	return (vol_state, rev_vol_state)

'''
@jax.jit
def my_where(x: jax.Array):
	# Elementwise comparison without jnp.where, using list comprehension
	return jnp.array([i if i >= 1E-12 else 1E-12 for i in x])
'''

###### functions to replace jnp.where with lax.cond for vectorized operations ######
@jax.jit
def my_where(x: jax.Array):
	def scalar_where(i):
		return jax.lax.cond(i >= 1e-12, lambda _: i, lambda _: 1e-12, operand=None)
	return jax.vmap(scalar_where)(x)

@jax.jit
def my_stretch_where(ref_mag_state, exten_state):
    # r, e both shape (num_nodes, num_neighbors)
	
    return jnp.where(ref_mag_state > 1e-16, exten_state / ref_mag_state, 0.0)


@jax.jit
def inf_state_where(inf_state: jax.Array, stretch: jax.Array, critical_stretch: float):
	"""
	Zero out inf_state where stretch > critical_stretch, using lax.cond and double vmap.
	Works for 2D arrays (nodes x neighbors) or 1D arrays.
	"""
	# Scalar conditional
	def cond_fn(s, i):
		return jax.lax.cond(s > critical_stretch, lambda _: 0.0, lambda _: i, operand=None)

	# Vectorize over columns (neighbors)
	row_fn = jax.vmap(cond_fn, in_axes=(0, 0))
	# Vectorize over rows (nodes)
	return jax.vmap(row_fn, in_axes=(0, 0))(stretch, inf_state)

@jax.jit
def my_replace_zero_val_where(x_array: jax.Array, eps: float):
	"""
	Replace zeros in x_array with eps, using lax.cond and double vmap.
	Works for 1D or 2D arrays.
	"""
	# Scalar conditional
	def cond_fn(i):
		return jax.lax.cond(i == 0.0, lambda _: eps, lambda _: i, operand=None)

	# Vectorize over columns
	row_fn = jax.vmap(cond_fn, in_axes=0)
	# Vectorize over rows
	return jax.vmap(row_fn, in_axes=0)(x_array)

@jax.jit
def shape_tens_eps_where(shape_tens: jax.Array, epsilon: float):
	def cond_fn(val):
		return jax.lax.cond(jnp.abs(val) < epsilon, lambda _: epsilon, lambda _: val, operand=None)
	return jax.vmap(cond_fn)(shape_tens)

	
### to wrap jnp.where to try to get rid of nans in grad
@jax.jit
def safe_divide(num: jax.Array, denom: jax.Array, eps: float = 1e-12):
	"""
	Safely divide num / denom, replacing small denom values with 1.0
	using lax.cond and double vmap for vectorized arrays.
	"""

	# Scalar conditional
	def cond_fn(d):
		return jax.lax.cond(jnp.abs(d) < eps, lambda _: 1.0, lambda _: d, operand=None)

	# Vectorize over columns
	row_fn = jax.vmap(cond_fn, in_axes=0)
	# Vectorize over rows
	denom_safe = jax.vmap(row_fn, in_axes=0)(denom)

	return num / denom_safe


@jax.jit
def compute_stable_time_step(families, ref_mag_state, volumes, num_nodes, 
        bulk_modulus,rho,horizon):

    spring_constant = 18.0 * bulk_modulus / math.pi / horizon**4.0

    crit_time_step_denom = np.array([spring_constant * volumes[families[i]] / 
            ref_mag_state[i] for i in range(num_nodes)])**0.5

    critical_time_steps = np.sqrt(2.0 * rho) / crit_time_step_denom
    
    nodal_min_time_step = [ np.amin(item) for item in critical_time_steps ]

    return np.amin(nodal_min_time_step)


###########################

# Compute the force vector-state using a LPS peridynamic formulation
@partial(jit, static_argnums=(4,))
def compute_force_state_LPS(params,disp:jax.Array,  vol_state:jax.Array, inf_state:jax.Array, allow_damage: bool) -> Tuple[jax.Array, jax.Array]:
		
	#Define some local convenience variables     
	ref_pos = params.pd_nodes 
	ref_pos_state = params.reference_position_state
	ref_mag_state = params.reference_magnitude_state
	neigh = params.neighborhood
	K = params.bulk_modulus
	critical_stretch = params.critical_stretch
	no_damage_region_left = params.no_damage_region_left
	no_damage_region_right = params.no_damage_region_right
	undamaged_influence_state_left = params.undamaged_influence_state_left
	undamaged_influence_state_right = params.undamaged_influence_state_right
 

	#disp = 0.001
	#disp =  disp[0]
	#jnp.zeros_like(ref_pos)
	#jax.debug.print("critical_stretch = {c}", c=critical_stretch)

	#Compute the deformed positions of the nodes
	def_pos = ref_pos + disp
	#jax.debug.print("disp: {d}", d=disp)

	#Compute deformation state
	def_state = def_pos[neigh] - def_pos[:,None]
	#jax.debug.print("def_state finite: {b}", b=jnp.all(jnp.isfinite(def_state)))
	#jax.debug.print("def_state zeros? {z}", z=jnp.any(def_state == 0))

	
	# Compute deformation magnitude state
	#def_mag_state = jnp.sqrt(def_state * def_state)
	#def_mag_state = jnp.linalg.norm(def_state, axis=-1)
	#def_mag_state = (def_state * def_state) ** 0.5
	def_mag_state = jnp.sqrt(jnp.maximum(def_state * def_state, 1e-24))
	#jax.debug.print("def_mag_state: {b}", b=def_mag_state)


	# Compute deformation unit state
	eps = 1e-10
	def_unit_state = def_state / (def_mag_state + 1e-12)
	#def_unit_state = jnp.where(def_mag_state > 1.0e-12, def_state / def_mag_state, 0.0)
	#def_unit_state = my_stretch_where(def_mag_state[..., None], def_state)
	def_unit_state = jnp.where(def_mag_state > eps,
							def_state / def_mag_state, 0.0)
	#def_unit_state = jax.vmap(safe_unit)(def_state, def_mag_state)


	# Compute scalar extension state
	exten_state = def_mag_state - ref_mag_state
 	#exten_state = def_mag_state[:, None] - ref_mag_state

	#exten_state = def_mag_state - ref_mag_state
	#jax.debug.print("[exten_state] min={m} max={M}",  m=jnp.min(exten_state), M=jnp.max(exten_state))

	## switched out for jnp where OG code to see if 
	stretch = jnp.where(ref_mag_state > 1.0e-16, exten_state / ref_mag_state, 0.0)
	#jax.debug.print("stretch beofore my_stretch_where {s}", s=exten_state / ref_mag_state)
	#stretch = my_stretch_where(ref_mag_state, exten_state)
	
    # Apply critical stretch fracture criteria to update inf state
	def damage_branch(inf_state):
		#inf_state_updated = jnp.where(stretch > critical_stretch, 0.0, inf_state)
		#inf_state = inf_state_where(inf_state, stretch, critical_stretch)
		inf_state = jnp.where(stretch > critical_stretch, 0.0, inf_state)
		#jax.debug.print("inf_state after inf_state_where {i}", i=inf_state)
		inf_state = inf_state.at[no_damage_region_left, :].set(undamaged_influence_state_left)
		inf_state = inf_state.at[no_damage_region_right, :].set(undamaged_influence_state_right)
		return inf_state

	def no_damage_branch(inf_state):
		# return state unchanged
		return inf_state


  
	# Apply a critical strech fracture criteria
	inf_state_updated = lax.cond(allow_damage, damage_branch, no_damage_branch, inf_state)
	#jax.debug.print("inf_state after damage/no_damage_branch {i}", i=inf_state)

	#jax.debug.print("[inf_state pre-eps] any==0? {z} finite={f}",
				#z=jnp.any(inf_state == 0.0), f=jnp.all(jnp.isfinite(inf_state)))

	eps = 1e-10  # or smaller if your scale is tiny

	#inf_state = jnp.where(inf_state == 0.0, eps, inf_state)
	inf_state = my_replace_zero_val_where(inf_state, eps)

	#ref_pos_state = jnp.where(ref_pos_state == 0.0, eps, ref_pos_state)
	ref_pos_state = my_replace_zero_val_where(ref_pos_state, eps) 
	#jax.debug.print("ref_pos_state zeros? {z}", z=jnp.any(ref_pos_state == 0))
	
	# Compute the shape tensor (really a scalar because this is 1d), i.e. the "weighted volume" as 
	# defined in Silling et al. 2007
	# added epsilon to prevent dividing by zero
	epsilon = 1e-8
	shape_tens = (inf_state * ref_pos_state * ref_pos_state * vol_state).sum(axis=1)
	#shape_tens = jnp.where(jnp.abs(shape_tens) < epsilon, epsilon, shape_tens)
	#shape_tens = shape_tens_eps_where(shape_tens, epsilon)
	shape_tens = jnp.maximum(shape_tens, epsilon)
	

	# Compute scalar force state for a elastic constitutive model
	######### compute strain energy density here?  or calculation at least should look like this line here ########
	#scalar_force_state = 9.0 * K / shape_tens[:, None] * exten_state
	scalar_force_state = 9.0 * K * safe_divide(exten_state, shape_tens[:, None])

	# bond strain energy calc
	#bond_strain_energy = 9.0 * K / shape_tens[:, None] * exten_state * exten_state * ref_mag_state
	bond_strain_energy = 9.0 * K * safe_divide(exten_state**2 * ref_mag_state, shape_tens[:, None])

	# Compute the force state
	force_state = inf_state * scalar_force_state * def_unit_state
	#jax.debug.print("force_state finite: {b}", b=force_state)

	###  return bond_strain_energy
	return force_state, inf_state_updated, bond_strain_energy



def smooth_ramp(t, t0, c=1.0, beta=5.0):
	"""
	Function that linearly ramps up to c at t0, then smoothly transitions to c.

	Parameters:
	- t: Time variable (scalar or numpy array).
	- t0: Time at which the transition occurs.
	- c: Final constant value after transition.
	- beta: Smoothness parameter (higher values = sharper transition).

	Returns:
	- f: Value of the function at time t.
	"""
	# Linear ramp before t0 (with slope c/t0)
	linear_ramp = (c / t0) * t

	# Smooth transition using an exponential decay term
	smooth_transition = c * (1 - jnp.exp(-beta * (t - t0))) + (c / t0) * t0

	# Use `jnp.where` to define the piecewise function
	f = jnp.where(t < t0, linear_ramp, smooth_transition)
 
	
	return f

@jax.jit
def save_if_needed(forces_array, force_value, step_number):
    """
    Save force_value into forces_array based on step_number rules:
      - every step when step_number < 100
      - every 500 steps when 100 <= step_number < 2000
      - every 5000 steps when step_number >= 2000
    """
    mask = jnp.logical_or(
        # Phase 1: 0-6000, every 500 steps
        jnp.logical_and(step_number <= 6000, step_number % 500 == 0),

        # Phase 2: 6000-15000, every 1000 steps
        jnp.logical_and(
            step_number >= 6000,
            jnp.logical_and(
                step_number <= 15000,
                step_number % 1000 == 0
            )
        )
    )

    # Use lax.cond to choose branch without Python-side branching
    def write(arr):
        return arr.at[step_number].set(force_value)
    def skip(arr):
        return arr
    return jax.lax.cond(mask, write, skip, forces_array)

@jax.jit
def calc_damage_if_needed(vol_state, inf_state, undamaged_inf_state, damage, step_number, force_value):
    """
    If step_number is one we want to save, write force_value at that index,
    otherwise return forces_array unchanged.
    """
    mask = jnp.logical_or(
        # Phase 1: 0-6000, every 500 steps
        jnp.logical_and(step_number <= 6000, step_number % 500 == 0),

        # Phase 2: 6000-15000, every 1000 steps
        jnp.logical_and(
            step_number >= 6000,
            jnp.logical_and(
                step_number <= 15000,
                step_number % 1000 == 0
            )
        )
    )

	
    # Use lax.cond to choose branch without Python-side branching
    
    def write(arr):
        damage = compute_damage(vol_state, inf_state, undamaged_inf_state)
        return arr.at[step_number].set(damage)
    def skip(arr):
        return arr
    return jax.lax.cond(mask, write, skip, damage)

@jax.jit
def save_disp_if_needed(disp_array, disp_value, step_number):
    """
    If step_number is one we want to save, write disp_value at that index,
    otherwise return disp_array unchanged.
    """
    mask = jnp.logical_or(
        # Phase 1: 0-6000, every 500 steps
        jnp.logical_and(step_number <= 6000, step_number % 500 == 0),

        # Phase 2: 6000-15000, every 1000 steps
        jnp.logical_and(
            step_number >= 6000,
            jnp.logical_and(
                step_number <= 15000,
                step_number % 1000 == 0
            )
        )
    )

    # Use lax.cond to choose branch without Python-side branching
    def write(arr):
        return arr.at[step_number].set(disp_value)
    def skip(arr):
        return arr
    return jax.lax.cond(mask, write, skip, disp_array)

@jax.jit
def save_velo_if_needed(velo_array, velo_value, step_number):
    """
    If step_number is one we want to save, write velo_value at that index,
    otherwise return velo_array unchanged.
    """
    mask = jnp.logical_or(
        # Phase 1: 0-6000, every 500 steps
        jnp.logical_and(step_number <= 6000, step_number % 500 == 0),

        # Phase 2: 6000-15000, every 1000 steps
        jnp.logical_and(
            step_number >= 6000,
            jnp.logical_and(
                step_number <= 15000,
                step_number % 1000 == 0
            )
        )
    )

    # Use lax.cond to choose branch without Python-side branching
    def write(arr):
        return arr.at[step_number].set(velo_value)
    def skip(arr):
        return arr
    return jax.lax.cond(mask, write, skip, velo_array)

# Internal force calculation
#@jax.jit(static_argnums=7)  
@partial(jax.jit, static_argnums=(14))
def compute_internal_force(params, disp, vol_state, rev_vol_state, inf_state, undamaged_inf_state, damage, thickness, time, time_step, num_steps, forces_array, disp_array, step_number, allow_damage):

	# Define some local convenience variables     
	neigh = params.neighborhood
	prescribed_force = params.prescribed_force
	width = params.width
	num_nodes = params.num_nodes
	left_bc_region = params.left_bc_region
	right_bc_region = params.right_bc_region

	#jax.debug.print("disp zeros? {z}", z=jnp.any(disp == 0))
	

	##### return bond_strain_energy #####
	force_state, inf_state, bond_strain_energy = compute_force_state_LPS(params, disp, vol_state, inf_state, allow_damage)

	#Integrate nodal forces 
	force = (force_state * vol_state).sum(axis=1)
	force = force.at[neigh].add(-force_state * rev_vol_state)

	#strain_energy = bond_strain_energy
	strain_energy = 0.5 * (bond_strain_energy * vol_state).sum(axis=1)

	#total_strain_energy = jnp.sum(strain_energy)
	#jax.debug.print("strain_energy: {s}", s=jnp.sum(strain_energy))

	if prescribed_force is not None:
		li = 0
		ri = num_nodes - 1
		#ramp_force = smooth_ramp(time, t0=1.e-3, c=prescribed_force) 
		ramp_force = smooth_ramp(time, t0=1.e-5, c=prescribed_force) 



		denom_left  = vol_state[li].sum() + rev_vol_state[li][0]
		denom_right = vol_state[ri].sum() + rev_vol_state[ri][0]



		eps = 1e-12
		#denom_left  = jnp.where(jnp.abs(denom_left)  < eps, eps, denom_left)
		#denom_right = jnp.where(jnp.abs(denom_right) < eps, eps, denom_right)

		denom_left  = jnp.clip(denom_left,  1e-3, jnp.inf)
		denom_right = jnp.clip(denom_right, 1e-3, jnp.inf)


		# Compute the left boundary condition nodal forces
		left_bc_area = width * thickness[left_bc_region]
		#left_bc_nodal_forces = (ramp_force * left_bc_area)/(vol_state[li].sum() + rev_vol_state[li][0])
		left_bc_nodal_forces = (ramp_force * left_bc_area)/denom_left
		force = force.at[left_bc_region].add(-left_bc_nodal_forces)

		# For the leftmost node (if needed)
		left_bc_area_li = width * thickness[li]
		force = force.at[li].add(-ramp_force * left_bc_area_li)

		# Compute the right boundary condition nodal forces
		right_bc_area = width * thickness[right_bc_region]
		#right_bc_nodal_forces = (ramp_force * right_bc_area)/(vol_state[ri].sum() + rev_vol_state[ri][0])
		right_bc_nodal_forces = (ramp_force * right_bc_area)/denom_right
		force = force.at[right_bc_region].add(right_bc_nodal_forces)

		# For the rightmost node (if needed)
		right_bc_area_ri = width * thickness[ri]
		force = force.at[ri].add(ramp_force * right_bc_area_ri)

	forces_array = save_if_needed(forces_array, ramp_force, step_number)
	damage = calc_damage_if_needed(vol_state, inf_state, undamaged_inf_state, damage, step_number, ramp_force)

	return force, inf_state, strain_energy, ramp_force, forces_array, damage


@partial(jax.jit, static_argnums=(2,))
def solve_one_step(params, vals, allow_damage:bool):

	(disp, vel, acc, vol_state, rev_vol_state, inf_state, thickness, undamaged_inf_state, damage, forces_array, disp_array, velo_array, strain_energy, time) = vals
	prescribed_velocity = params.prescribed_velocity
	bar_length = params.bar_length
	left_bc_region = params.left_bc_region
	right_bc_region = params.right_bc_region
	pd_nodes = params.pd_nodes
	rho = params.density
	E = params.elastic_modulus 
	horizon = params.horizon
	families = params.neighborhood
	ref_mag_state = params.reference_magnitude_state



	# TODO: Solve for stable time step
 	#time_step = compute_stable_time_step(families, ref_mag_state, volumes, num_nodes, bulk_modulus, rho, horizon)
	#jax.debug.print("Computed stable time step: {t}", t=time_step)
	#time_step = 5.0E-08
	#time_step = 2.5E-08
	#time_step = 4.75E-08
	time_step = 2.8E-08

	num_steps = max_time / time_step

	#jax.debug.print("in solve_one_step: {t}", t=time)
	##########################
	# Check inputs for NaNs
	##########################

	# Check if any of the input arrays contain NaNs
	for name, vals in zip(["disp", "vel", "acc", "vol_state", "rev_vol_state", "inf_state", "thickness", "strain_energy"], [disp, vel, acc, vol_state, rev_vol_state, inf_state, thickness, strain_energy]):
		is_finite = jnp.all(jnp.isfinite(vals))
		#jax.debug.print("no NaNs detected in {n}: {f}", n=vals, f=is_finite)

	if prescribed_velocity is not None:
		bc_value = prescribed_velocity * time
		# Apply displacements bcs
		f = lambda x: 2.0 * bc_value / bar_length * x
		disp = disp.at[left_bc_region].set(f(pd_nodes[left_bc_region]))
		disp = disp.at[right_bc_region].set(f(pd_nodes[right_bc_region]))

	step_number = jnp.floor_divide(time, time_step).astype(jnp.int32)

	force, inf_state, strain_energy, ramp_force, forces_array, damage = compute_internal_force(params, disp, vol_state, rev_vol_state, inf_state, undamaged_inf_state, damage, thickness, time, time_step, num_steps, forces_array, disp_array, step_number, allow_damage)

	acc_new = force / rho

	vel = vel.at[:].add(0.5 * (acc + acc_new) * time_step)

	disp = disp.at[:].add(vel * time_step + (0.5 * acc_new * time_step * time_step))
	acc = acc.at[:].set(acc_new)
	
	#step_number = time / time_step
	step_number = jnp.floor_divide(time, time_step).astype(int)
	disp_array = save_disp_if_needed(disp_array, disp, step_number)
 
	velo_array = save_velo_if_needed(velo_array, vel, step_number)

	#damage_updated = compute_damage(vol_state, inf_state, undamaged_inf_state)

	#jax.debug.print("disp in solve_one_step: {d}", d=disp)

	### calculating strain energy density
	#strain_energy_total = jnp.sum(strain_energy)
	#total_volume = jnp.sum(vol_state)
	#strain_energy_density = strain_energy_total / total_volume

	#strain_energy_L1_norm = jnp.linalg.norm(strain_energy, ord=1, axis=0)

	#jax.debug.print("strain_energy_total in solve: {s}", s=strain_energy_total)

	def nan_debug_print(x):
		jax.debug.print("NaNs detected in strain_energy_total")
		return x

	def no_nan(x):
		return x


	#jax.debug.print("disp: {d}", d=disp)
	#jax.debug.print("disp? {z}", z=jnp.any(disp == 0))

	return (disp, vel, acc, vol_state, rev_vol_state, inf_state, thickness, undamaged_inf_state, damage, forces_array, disp_array, velo_array, strain_energy, time + time_step)


### put wrapper on solve
#@partial(jax.jit, static_argnums=(3,4))
def _solve(params, state, thickness:jax.Array, forces_array:jax.Array, allow_damage:bool, max_time:float=1.0):
	
	#Solves in time using Verlet-Velocity integration

    EPS = 1.0e-12  # Minimum safe volume to avoid NaNs

    #time_step = 2.5E-08
    #time_step = 4.75E-08
    time_step = 2.8E-08

    num_steps = int(max_time / time_step)

    vol_state, rev_vol_state = compute_partial_volumes(params, thickness)
    
    # Reset forces array
    forces_array = jnp.full((num_steps,), 0.0)

	# Clamp to avoid divide-by-zero or log(0) NaNs
    vol_state = jnp.maximum(vol_state, EPS)
    rev_vol_state = jnp.maximum(rev_vol_state, EPS)

    #jax.debug.print("vol_state= {V}",V=vol_state)
    #jax.debug.print("vol_state min={vmin}", vmin=jnp.min(vol_state))
    #jax.debug.print("vol_state zeros? {z}", z=jnp.any(vol_state == 0))


    inf_state = state.influence_state.copy() 
    undamaged_inf_state = state.undamaged_influence_state.copy()

	# Initialize a fresh influence state for this run
    #inf_state = jnp.where(vol_state > 1.0e-16, 1.0, 0.0)
    #jax.debug.print("inf_state after init: {i}",i=inf_state)

	#jax.debug.print("inf_state update after where: {i}",i=inf_state)
	# The fields
    disp = jnp.zeros_like(params.pd_nodes)
    vel = jnp.zeros_like(params.pd_nodes)
    acc = jnp.zeros_like(params.pd_nodes)
    time = 0.0
    #strain_energy = 0.0
    
    strain_energy = jnp.zeros_like(params.pd_nodes)
    #damage = jnp.zeros_like(params.pd_nodes)
    damage = jnp.zeros((num_steps, params.num_nodes))
    disp_array = jnp.zeros((num_steps, params.num_nodes))
    velo_array = jnp.zeros((num_steps, params.num_nodes))

    #jax.debug.print("Damage after reset: {d}", d=damage)

    def loop_body(i, vals):
        new_vals = solve_one_step(params, vals, allow_damage)
        return new_vals

	#Solve
    vals = (disp, vel, acc, vol_state, rev_vol_state, inf_state, thickness, undamaged_inf_state, damage, forces_array, disp_array, velo_array, strain_energy, time)

    vals_returned = jax.lax.fori_loop(0, num_steps, loop_body, vals)
    
    #jax.debug.print("Damage after sim: {d}", d=vals_returned[9])

    # Using mask to save forces at desired steps for plotting animation
    step_inds = jnp.arange(num_steps)
    
    mask_all = jnp.logical_or(
        # Phase 1: 0-6000, every 500 steps
        jnp.logical_and(step_inds <= 6000, step_inds % 500 == 0),

        # Phase 2: 6000-15000, every 1000 steps
        jnp.logical_and(
            step_inds >= 6000,
            jnp.logical_and(
                step_inds <= 15000,
                step_inds % 1000 == 0
            )
        )
    )


    forces_saved = vals_returned[9][mask_all]
    #jax.debug.print("forces_saved after sim: {f}", f=forces_saved)
    #jax.debug.print("forces_saved after sim: {f}", f=forces_saved.shape)

    damage_saved = vals_returned[8][mask_all]
    #jax.debug.print("damage_saved after sim: {d}", d=damage_saved)
    #jax.debug.print("damage_saved after sim: {d}", d=damage_saved.shape)
    
    
    disp_saved = vals_returned[10][mask_all]

    vel_saved =  vals_returned[11][mask_all]

    return PDState(disp=vals_returned[0], vel=vals_returned[1], acc=vals_returned[2], vol_state=vals_returned[3], rev_vol_state=vals_returned[4], influence_state=vals_returned[5],
                   undamaged_influence_state=vals_returned[7], damage=damage_saved, forces_array=forces_saved, disp_array=disp_saved, velo_array=vel_saved, strain_energy=vals_returned[12], time=vals_returned[13])
### put wrapper on solve
#@partial(jax.jit, static_argnums=(3,4))
def smooth_ramp(t, t0, c=1.0, beta=5.0):
	"""
	Function that linearly ramps up to c at t0, then smoothly transitions to c.

	Parameters:
	- t: Time variable (scalar or numpy array).
	- t0: Time at which the transition occurs.
	- c: Final constant value after transition.
	- beta: Smoothness parameter (higher values = sharper transition).

	Returns:
	- f: Value of the function at time t.
	"""
	# Linear ramp before t0 (with slope c/t0)
	linear_ramp = (c / t0) * t

	# Smooth transition using an exponential decay term
	smooth_transition = c * (1 - jnp.exp(-beta * (t - t0))) + (c / t0) * t0

	# Use `np.where` to define the piecewise function
	f = jnp.where(t < t0, linear_ramp, smooth_transition)
	
	return f

def smooth_ramp_centered(t, t0, c=1.0, beta=5.0, nodes=None, bar_length=None):
    """Smooth ramp applied as a Gaussian centered force distribution."""
    ramp = c * (1 - jnp.exp(-beta * jnp.maximum(0.0, t - t0)))
    # spatial distribution: peak at center, decays to ends
    if nodes is not None and bar_length is not None:
        x = nodes - bar_length/2.0
        sigma = bar_length/8.0  # controls spread
        spatial = jnp.exp(-(x**2)/(2*sigma**2))
        spatial /= spatial.sum()  # normalize
        return ramp * spatial
    else:
        return ramp

def ensure_thickness_vector(thickness, num_nodes):
	# Accept float, scalar array, or vector. Return vector of length num_nodes (NumPy/JAX array).
	thickness = jnp.asarray(thickness)
	if thickness.ndim == 0 or thickness.size == 1:
		return jnp.full((num_nodes,), float(thickness))
	if thickness.shape != (num_nodes,):
		raise ValueError(f"Thickness must have shape ({num_nodes},), got {thickness.shape}")
	return thickness

@jax.jit
def zero_small_inf_state(inf_state: jax.Array, threshold: float = 1e-9) -> jax.Array:
    """
    Set entries in inf_state to zero wherever they are less than threshold.
    Works for 1D or 2D arrays.
    """
    def cond_fn(val):
        return jax.lax.cond(val < threshold, lambda _: 0.0, lambda _: val, operand=None)
    # Vectorize over columns
    row_fn = jax.vmap(cond_fn, in_axes=0)
    # Vectorize over rows
    return jax.vmap(row_fn, in_axes=0)(inf_state)

@jax.jit
def thickness_in_no_damage_region(thickness:jax.Array, no_damage_region_left:jax.Array, no_damage_region_right:jax.Array, min_thickness:float=1.0):
    """
    Set thickness to min_thickness in the no-damage regions.
    """
    thickness = thickness.at[no_damage_region_left].set(min_thickness)
    thickness = thickness.at[no_damage_region_right].set(min_thickness)
    return thickness

def compute_damage(vol_state:jax.Array, inf_state:jax.Array, undamaged_inf_state:jax.Array):
	# returning small inf_state values to zero
	inf_state = zero_small_inf_state(inf_state)
	#jax.debug.print("vol_state in comp damage: {i}", i=vol_state)
	return 1 - ((inf_state * vol_state).sum(axis=1)) / ((undamaged_inf_state * vol_state).sum(axis=1))

def loss(params, state, thickness_vector:Union[float, jax.Array], forces_array:Union[float, jax.Array], allow_damage:bool, max_time:float):
	# thickness_vector = thickness_vector[0] * jnp.ones(params.num_nodes)
 
 	# Thickness can't change in no_damage regions
	min_thickness_no_damage = 5.0
	#thickness_vector = thickness_in_no_damage_region(thickness_vector, params.no_damage_region_left, params.no_damage_region_right, min_thickness_no_damage)


	output_vals = _solve(params, state, thickness=thickness_vector, forces_array=forces_array, allow_damage=allow_damage, max_time=max_time)
	checkify.check(jnp.all(jnp.isfinite(output_vals[0])), "NaN in solution")
	
	jax.debug.print("inf_state after run: {i}", i=output_vals[5])

	# Extract strain energy density from output_vals
	strain_energy = output_vals[11]
 
	# Extract damage from output_vals
	damage = output_vals[7][-1]

	# Calc strain energy density
	#total_volume = jnp.sum(vol_state)
	#strain_energy_density = strain_energy_total / total_volume

	# calc L1 norm strain energy density
	#strain_energy_L1_norm = jnp.linalg.norm(strain_energy, ord=1, axis=0)
	strain_energy_norm = jnp.linalg.norm(strain_energy, ord=jnp.inf)

	normalization_factor = 1E12
	
	mean_thickness = thickness_vector.mean()
	max_thickness = thickness_vector.max()
	min_thickness = thickness_vector.min()

	critical_thickness = 0.5
	penalty_weight = 1e15
	thickness_penalty = jnp.maximum(0.0, critical_thickness - min_thickness)

	# loss value that implements thickness penalty
	# loss_value = strain_energy_norm / normalization_factor 
	# loss_value = strain_energy_norm / normalization_factor + mean_thickness/max_thickness * 1E9 


	#print("undamaged_inf_state: ", output_vals[6])

	#Calling compute damage 
	#jax.debug.print("inf state: {i}", i=output_vals[5])
	loss_value = damage.sum()

	jax.debug.print("damage: {l}", l=damage)
	#loss_value = jnp.linalg.norm(damage, ord=1)
	#loss_value = jnp.linalg.norm(damage, ord=2) * 10.0
	#loss_value = damage.mean() * 1.0E3
 
    # Weights for balancing mean and max (adjust w_mean and w_max based on your priorities)
	#w_sum = 0.5 # Weight for mean damage (e.g., 70% focus on overall average)
	#w_max = 0.5  # Weight for max damage (e.g., 30% focus on peaks)

	#sum_damage = damage.sum()  #
	#max_damage = jnp.max(damage)    # Maximum damage value across all nodes and saved time steps

	#loss_value = w_sum * sum_damage + (w_max * max_damage * 10.0)


	#### Analyzing different loss functions ####
	# No thickness dependence
	# loss_value = strain_energy_norm / normalization_factor

	# With thickness dependence, penalize for total thickness
	#loss_value = strain_energy_density / normalization_factor + thickness_vector.sum()
	
	# Ratio, encourage low strain energy per unit thickness 
	#loss_value = (strain_energy_density / normalization_factor) / (jnp.sum(thickness_vector) + 1e-8)

	# Max thickness penalization, limit largest thickness
	gamma = 100.0
	#loss_value =  strain_energy_density / normalization_factor + gamma * jnp.max(thickness_vector)

	# Inverse thickness penalization, prefer thicker regions for strength
	#loss_value = jnp.sum(strain_energy_density / (thickness_vector + 1e-8))

	# Weighted combination, multi-objective
	#loss_value = 0.6 * (strain_energy_norm / normalization_factor) + 0.4 * (jnp.sum(thickness_vector) / normalization_factor)

	return loss_value, (strain_energy, damage)



### Main Program ####
if __name__ == "__main__":
    # Define fixed parameters
    fixed_length = 10.0  # Length of the bar
    delta_x = 0.11       # Element length
    fixed_horizon = 3.6 * delta_x  # Horizon size
    thickness = 1.0  # Thickness of the bar
    num_elems = int(fixed_length/delta_x)
    
    # material propertie selected for copper 
    #density = 8960.0
    #elastic_modulus = 130E9
    #mode1_fracture_tough = 76.0E6  # Mode I fracture toughness in J/m^2
    #poisson_ratio = 0.3
    
    #prescribed_force = 1.0E1
	
	# material properties selected for 304 stainless steel
    density = 7930.0
    elastic_modulus = 200E9
    mode1_fracture_tough = 120.0E6  # Mode I fracture toughness in J/m^2
    poisson_ratio = 0.34
    prescribed_force = 5.0E9


    bulk_modulus = elastic_modulus / (3 * (1 - 2 * poisson_ratio))
    G = mode1_fracture_tough ** 2 / elastic_modulus  # Critical strain energy release rate
    
    #prescribed_velocity = 0.01  # Prescribed velocity at the boundaries

    
    # Compute critical stretch from fracture toughness as done in Silling and Askari 2005
    # Now critical stretch adjusts based on horizon, discretization, and material properties
    critical_stretch = np.sqrt(5 * G / (9 * bulk_modulus * fixed_horizon))
    #Critical stretch for damage, set to None for no damage, 1e-4
    #critical_stretch = 1.0e-04
    
    print("critical_stretch: ", critical_stretch)

    allow_damage = True

    '''
    if critical_stretch is None:
        critical_stretch = 1.0e3
        allow_damage = False   # Large value, effectively disabling damage

    if critical_stretch is not None:
        allow_damage = True
    '''
    #key = jax.random.PRNGKey(np.random.randint(0, 1_000_000))  # Use a random seed each run
    #shape = (num_elems,)  # Example shape
    #thickness = jax.random.uniform(key, shape=shape, minval=0.5, maxval=1.5)
	

    #key = jax.random.PRNGKey(7)  # Seed for reproducibility
    #shape = (num_elems,)  # Example shape
    #thickness = jax.random.uniform(key, shape=shape, minval=0.5, maxval=1.5)

    #thickness = jnp.full((num_elems,), 1.0)
    #scalar_param = 0.5
    #hickness0 = scalar_param * jnp.ones(num_elems)
    #thickness =  1.0
    thickness0 = thickness 

    

    # Initialize the problem with fixed parameters
    params, state = init_problem(
        bar_length=fixed_length,
        density=density,
        bulk_modulus=bulk_modulus,
        elastic_modulus=elastic_modulus,
        number_of_elements=int(fixed_length / delta_x),
        horizon=fixed_horizon,
        thickness=thickness,
        prescribed_force=prescribed_force,
        critical_stretch= critical_stretch)
	
    #max_time = 1.0E-01
    #max_time = 1.0
    max_time = 1.0E-02
    
    max_time = float(max_time)
    
    
    time_step = 2.5E-08
    num_steps = int(max_time / time_step)
    
    forces_array = jnp.full((num_steps,), 0.0)
    
    #key = jax.random.PRNGKey(0)  # Seed for reproducibility
    #shape = (params.num_nodes,)  # Example shape
    #param = jax.random.uniform(key, shape=shape, minval=0.5, maxval=1.2)

    # Solve the problem with initial thickness
    thickness0 = ensure_thickness_vector(thickness0, num_elems)
    results = _solve(params, state, thickness0, forces_array=forces_array, allow_damage=allow_damage, max_time=float(max_time))
    jax.debug.print("allow_damage in main: {a}", a=allow_damage)
    
    print("thickness inputted: ", thickness0)
    
    #print("damage: ", results[7])
    
    #print("displacement values: ", results[0])
    disp = results[0]
    fig, ax = plt.subplots()
    ax.plot(params.pd_nodes, disp, 'k.')
    ax.set_xlabel("Node Position")
    ax.set_ylabel("Displacement")
    ax.set_title("Displacement vs Node Position (Forward Problem)")
    #ax.set_ylim(-2.5,2.5)
    plt.tight_layout()
    plt.show()
    print("thickness: ",thickness0)
    print("damage_final: ", results[7][-1])
    
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
