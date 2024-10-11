#!/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
import scipy.spatial
import scipy.optimize
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit

from typing import Union, Tuple


class PDJAX():
    '''
       This class initializes a 1D peridynamic problem.  This problem is essentially
       a long bar with displacement boundary conditions applied to boundary regions
       equal to 1-horizon at each end.

       Initialization parameters are as follows:

       + ``bar_length`` - length of bar, it will be centered at the origin

       + ``number_of_elements`` - the discretization level

       + ``bulk_modulus``

       + ``density``

       + ``thickness``
    '''

    def __init__(self, 
                 bar_length:float=20,
                 number_of_elements:int=20,
                 bulk_modulus:float=100,
                 density:float=1.0,
                 thickness:Union[float, np.ndarray]=1.0,
                 horizon:Union[float, None]=None,
                 critical_stretch:Union[float, None] = None
                 ) -> None:
        '''
           Initialization function
        '''

        #Problem data
        self.bulk_modulus = bulk_modulus
        self.rho = density

        self.bar_length = bar_length
        self.number_of_elements = number_of_elements

        delta_x = bar_length / number_of_elements

        # This array contains the *element* node locations.  i.e., they define the discrete 
        # regions along the bar. The peridynamic node locations will be at the centroid of 
        # these regions.
        self.nodes = jnp.linspace(-bar_length / 2.0, bar_length / 2.0, num=number_of_elements + 1)

        # Set horizon from parameter list or as default
        if horizon != None:
            self.horizon = horizon
        else:
            self.horizon = delta_x * 3.015

        if isinstance(thickness, float):
            thickness = jnp.ones(number_of_elements) * thickness
        elif isinstance(thickness, np.ndarray):
            thickness = jnp.asarray(thickness)
            assert thickness.shape[0] == number_of_elements, ValueError("Thickness array length must match number of elements")

        # Compute the pd_node locations, kdtree, nns, reference_position_state, etc.
        self.setup_discretization(thickness)

        # Debug plotting
        # _, ax = plt.subplots()
        # self.line, = ax.plot(self.pd_nodes, self.displacement)
        self._allow_damage = False

        self.critical_stretch = critical_stretch

        if self.critical_stretch is not None:
            self.allow_damage() 

        return


    def allow_damage(self):
        self._allow_damage = True
        return


    def setup_discretization(self, thickness:np.ndarray):
        
        nodes = self.nodes

        # The lengths of the *elements*
        self.lengths = jnp.array(nodes[1:] - nodes[0:-1]) 

        # The PD nodes are the centroids of the elements
        self.pd_nodes = jnp.array(nodes[0:-1] + self.lengths / 2.0)
        self.num_nodes = self.pd_nodes.shape[0]

        # Create's a kdtree to do nearest neighbor search
        self.tree = scipy.spatial.cKDTree(self.pd_nodes[:,None])

        # Get PD nodes in the neighborhood of support + largest node spacing, this will
        # find all potential partial volume nodes as well. The distances returned from the
        # search turn out to be the reference_magnitude_state, so we'll store them now
        # to avoid needed to calculate later.
        reference_magnitude_state, neighborhood = self.tree.query(self.pd_nodes[:,None], 
                k=100, p=2, eps=0.0, distance_upper_bound=(self.horizon + np.max(self.lengths) / 2.0))


        self.num_neighbors = jnp.asarray((neighborhood != self.tree.n).sum(axis=1)) - 1
        self.max_neighbors = np.max((neighborhood != self.tree.n).sum(axis=1))

        # Convert to JAX arrays and trim down excess neighbors
        neighborhood = jnp.asarray(neighborhood[:, :self.max_neighbors])
        self.reference_magnitude_state = jnp.delete(reference_magnitude_state[:, :self.max_neighbors], 0,1)

        # Cleanup neighborhood
        row_indices = jnp.arange(neighborhood.shape[0]).reshape(-1, 1)
        neighborhood = jnp.where(neighborhood == self.tree.n, row_indices, neighborhood)
        self.neighborhood = jnp.delete(neighborhood,0,1)


        # Compute the reference_position_state.  Using the terminology of Silling et al. 2007
        self.reference_position_state = self.pd_nodes[self.neighborhood] - self.pd_nodes[:,None]

        # Compute the partial volumes
        self.compute_partial_volumes(thickness)

        # Cleanup reference_magnitude_state
        self.reference_magnitude_state = jnp.where(self.reference_magnitude_state == np.inf, 0.0, self.reference_magnitude_state)

        # A Numpy masked array containing the *influence vector-state* as defined in Silling et al. 2007
        # ratio = self.reference_magnitude_state / self.horizon
        # self.influence_state = jnp.ones_like(self.volume_state) - 35.0 * ratio ** 4.0 + 84.0 * ratio ** 5.0 - 70 * ratio ** 6.0 + 20 * ratio ** 7.0
        self.influence_state = jnp.where(self.volume_state > 1.0e-16, 1.0, 0.0)
        #
        # The fields
        self.displacement = jnp.zeros_like(self.pd_nodes)
        self.velocity = jnp.zeros_like(self.pd_nodes)
        self.acceleration = jnp.zeros_like(self.pd_nodes)


        return


    def compute_partial_volumes(self, thickness:np.ndarray):

        # Setup some local (to function) convenience variables
        neigh = self.neighborhood
        lens = self.lengths
        ref_mag_state = self.reference_magnitude_state
        horiz = self.horizon

        # Compute the volume_state, where the nodal volume = length * width * thickness, 
        # length calculation takes into account the partially #covered distances on either 
        # side of the horizon. 

        # Initialize the volume_state to the lengths * width * thickness
        width = 1.0
        vol_state_uncorrected = lens[neigh] * thickness[:, None] * width 

        # Place zeros in node locations that are not fully inside the support neighborhood nor have a partial volume
        vol_state = jnp.where(ref_mag_state < horiz + lens[neigh] / 2.0, vol_state_uncorrected, 0.0)

        # Check to see if the neighboring node has a partial volume
        is_partial_volume = jnp.abs(horiz - ref_mag_state) < lens[neigh] / 2.0

        # Two different scenarios:
        is_partial_volume_case1 = is_partial_volume * (ref_mag_state >= horiz)
        is_partial_volume_case2 = is_partial_volume * (ref_mag_state < horiz)

        # Compute the partial volumes conditionally
        vol_state = jnp.where(is_partial_volume_case1, (lens[neigh] / 2.0 - (ref_mag_state - horiz)) * width * thickness[:, None], vol_state)
        vol_state = jnp.where(is_partial_volume_case2, (lens[neigh] / 2.0 + (horiz - ref_mag_state)) * width * thickness[:, None], vol_state)

        # If the partial volume is predicted to be larger than the unocrrected volume, set it back
        # vol_state = jnp.where(vol_state > vol_state_uncorrected, vol_state_uncorrected, vol_state)
        vol_state = jnp.where(vol_state > vol_state_uncorrected, vol_state_uncorrected, vol_state)

        # Now compute the "reverse volume state", this is the partial volume of the "source" node, i.e. node i,
        # as seen from node j.  This doesn't show up anywhere in any papers, it's just used here for computational
        # convenience
        vol_array = lens[:,None] * width * thickness[:, None]
        rev_vol_state = jnp.ones_like(vol_state) * vol_array
        rev_vol_state = jnp.where(is_partial_volume_case1, (lens[:, None] / 2.0 - (ref_mag_state - horiz)) * width * thickness[:, None], rev_vol_state)
        rev_vol_state = jnp.where(is_partial_volume_case2, (lens[:, None] / 2.0 + (horiz - ref_mag_state)) * width * thickness[:, None], rev_vol_state)
        #If the partial volume is predicted to be larger than the uncorrected volume, set it back
        rev_vol_state = np.where(rev_vol_state > vol_array, vol_array, rev_vol_state)

        # Set attributes
        self.volume_state = vol_state
        self.reverse_volume_state = rev_vol_state

        return

    def introduce_flaw(self, location:float, allow_damage=False):

        if allow_damage:
            self.allow_damage()

        _, nodes_near_flaw = self.tree.query(location, k=self.max_neighbors, p=2, eps=0.0, 
                                             distance_upper_bound=(self.horizon + np.max(self.lengths)/2))

        # The search above will produce duplicate neighbor nodes, make them into a
        # unique 1-dimensional list
        nodes_near_flaw = np.array(np.unique(nodes_near_flaw), dtype=np.int64)
    
        # Remove the dummy entries
        nodes_near_flaw = nodes_near_flaw[nodes_near_flaw != self.tree.n]

        families = jnp.asarray(self.neighborhood)
        # Loop over nodes near the crack to see if any bonds in the nodes family
        # cross the crack path
        for idx in nodes_near_flaw:
            # Loop over node family
            node_family = families[idx][families[idx] != idx]
            for bond_idx, end_point_idx in enumerate(node_family):
                # Define the bond line segment as the line between the node and its
                # endpoint.
                min_node, max_node = np.sort([self.pd_nodes[idx], self.pd_nodes[end_point_idx]])

                if min_node < location and location < max_node:
                    self.influence_state = self.influence_state.at[idx, bond_idx].set(0.2)

        return

    # Compute the force vector-state using a LPS peridynamic formulation
    def compute_force_state_LPS(self, 
                                disp:jax.Array, 
                                inf_state:jax.Array) -> Tuple[jax.Array, jax.Array]:
            
        #Define some local convenience variables     
        ref_pos = self.pd_nodes 
        ref_pos_state = self.reference_position_state
        ref_mag_state = self.reference_magnitude_state
        vol_state = self.volume_state
        neigh = self.neighborhood
        K = self.bulk_modulus

        #Compute the deformed positions of the nodes
        def_pos = ref_pos + disp

        #Compute deformation state
        def_state = def_pos[neigh] - def_pos[:,None]

        # Compute deformation magnitude state
        def_mag_state = jnp.sqrt(def_state * def_state)

        # Compute deformation unit state
        def_unit_state = jnp.where(def_mag_state > 1.0e-16, def_state / def_mag_state, 0.0)

        # Compute scalar extension state
        exten_state = def_mag_state - ref_mag_state

        stretch = jnp.where(ref_mag_state > 1.0e-16, exten_state / ref_mag_state, 0.0)
        # jax.debug.print('stretch: {s}', s=stretch.at[:].get())

        # Apply a critical strech fracture criteria
        if self._allow_damage:
            inf_state = jnp.where(stretch  > self.critical_stretch, 0.0, inf_state)
            # Disallow damage on the boundary regions
            undamaged_influence_state_left = self.influence_state.at[self.no_damage_region_left, :].get()
            inf_state = inf_state.at[self.no_damage_region_left, :].set(undamaged_influence_state_left)
            undamaged_influence_state_right = self.influence_state.at[self.no_damage_region_right, :].get()
            inf_state = inf_state.at[self.no_damage_region_right, :].set(undamaged_influence_state_right)
            # jax.debug.print('IS: {infl}', infl=inf_state.at[:].get())

        # Compute the shape tensor (really a scalar because this is 1d), i.e. the "weighted volume" as 
        # defined in Silling et al. 2007
        shape_tens = (inf_state * ref_pos_state * ref_pos_state * vol_state).sum(axis=1)

        # Compute scalar force state for a elastic constitutive model
        scalar_force_state = 9.0 * K / shape_tens[:, None] * exten_state
       
        # Compute the force state
        force_state = inf_state * scalar_force_state * def_unit_state

        return force_state, inf_state

    def compute_damage(self):
        return 1 - self.influence_state.sum(axis=1) / self.num_neighbors


    def smooth_ramp(self, t, t0, c=1.0, beta=5.0):
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

   
    # Internal force calculation
    @partial(jit, static_argnums=(0,))
    def compute_internal_force(self, disp, inf_state, time):
            
        # Define some local convenience variables     
        vol_state = self.volume_state
        rev_vol_state = self.reverse_volume_state
        neigh = self.neighborhood
        
        # Compute the force vector-state according to the choice of constitutive
        # model  
        force_state, inf_state = self.compute_force_state_LPS(disp, inf_state)

        #Integrate nodal forces 
        force = (force_state * vol_state).sum(axis=1)
        force = force.at[neigh].add(-force_state * rev_vol_state)

        if self.prescribed_force is not None:
            li = 0
            ramp_force = self.smooth_ramp(time, t0=1.e-5, c=self.prescribed_force)
            left_bc_force_density = ramp_force / (vol_state[li].sum() + self.reverse_volume_state[li][0])
            left_bc_nodal_forces = left_bc_force_density * vol_state[li][self.left_bc_mask]
            force = force.at[self.left_bc_region].set(-left_bc_nodal_forces)
            force = force.at[li].set(-left_bc_force_density * self.reverse_volume_state[li][li])
            #
            ri = self.num_nodes - 1
            right_bc_force_density = ramp_force / (vol_state[ri].sum() + self.reverse_volume_state[ri][0])
            right_bc_nodal_forces = right_bc_force_density * vol_state[ri][self.right_bc_mask]
            force = force.at[self.right_bc_region].set(right_bc_nodal_forces)
            force = force.at[ri].set(right_bc_force_density * self.reverse_volume_state[ri][0])


        return force, inf_state


    def solve_one_step(self, vals:Tuple[jax.Array, jax.Array, jax.Array, jax.Array, float]):

        (disp, vel, acc, inf_state, time) = vals

        # TODO: Solve for stable time step
        time_step = 1.0e-9

        if self.prescribed_velocity is not None:
            bc_value = self.prescribed_velocity * time
            # Apply displacements bcs
            f = lambda x: 2.0 * bc_value / self.bar_length * x
            disp = disp.at[self.left_bc_region].set(f(self.pd_nodes[self.left_bc_region]))
            disp = disp.at[self.right_bc_region].set(f(self.pd_nodes[self.right_bc_region]))

        force, inf_state = self.compute_internal_force(disp, inf_state, time)

        acc_new = force / self.rho
        vel = vel.at[:].add(0.5 * (acc + acc_new) * time_step)
        disp = disp.at[:].add(vel * time_step + (0.5 * acc_new * time_step * time_step))
        acc = acc.at[:].set(acc_new)

        return (disp, vel, acc, inf_state, time + time_step)


    def solve(self, 
              prescribed_velocity:Union[float, None]=None, 
              prescribed_force:Union[float, None]=None, 
              max_time:float=1.0):
        '''
            Solves in time using Verlet-Velocity
        '''

        if prescribed_velocity is not None and prescribed_force is not None:
            raise ValueError("Only one of prescribed_velocity or prescribed_force should be set, not both.")
        
        if prescribed_velocity is None and prescribed_force is None:
            raise ValueError("Either prescribed_velocity or prescribed_force must be set.")

        self.prescribed_velocity = prescribed_velocity
        self.prescribed_force = prescribed_force

        #:The node indices of the boundary region at the left end of the bar
        li = 0
        self.left_bc_mask = self.neighborhood[li] != li
        self.left_bc_region = self.neighborhood[li][self.left_bc_mask]

        #:The node indices of the boundary region at the right end of the bar
        ri = self.num_nodes - 1
        self.right_bc_mask = self.neighborhood[ri] != ri
        self.right_bc_region = self.neighborhood[ri][self.right_bc_mask]

        if prescribed_velocity is not None:
            self.left_bc_region = jnp.asarray(self.tree.query_ball_point(self.pd_nodes[0, None], r=(self.horizon + np.max(self.lengths) / 2.0), p=2, eps=0.0)).sort()
            self.right_bc_region = jnp.asarray(self.tree.query_ball_point(self.pd_nodes[-1, None], r=(self.horizon + np.max(self.lengths) / 2.0), p=2, eps=0.0)).sort()

        self.no_damage_region_left = jnp.asarray(self.tree.query_ball_point(self.pd_nodes[0, None], r=(2.0*self.horizon + np.max(self.lengths) / 2.0), p=2, eps=0.0)).sort()
        #:The node indices of the boundary region at the right end of the bar
        self.no_damage_region_right = jnp.asarray(self.tree.query_ball_point(self.pd_nodes[-1, None], r=(2.0*self.horizon + np.max(self.lengths) / 2.0), p=2, eps=0.0)).sort()
        #Solve
        disp = self.displacement
        vel = self.velocity
        acc = self.acceleration
        inf_state = self.influence_state
        time = 0.0

        vals = (disp, vel, acc, inf_state, time)
        vals = jax.lax.while_loop(lambda vals: vals[4] < max_time, self.solve_one_step, vals)

        self.displacement = vals[0]
        self.velocity = vals[1]
        self.acceleration = vals[2]
        self.influence_state = vals[3]


        return

    def get_solution(self):
        ''' Convenience function for retrieving displacement'''
        return self.displacement

    def get_nodes(self):
        ''' Convenience function for retrieving peridynamic node locations'''
        return self.pd_nodes

def loss(thickness:jax.Array):

    fixed_length = 10.0 
    delta_x = 0.25
    fixed_horizon = 2.6 * delta_x
    num_elems = int(fixed_length/delta_x)

    problem1 = PDJAX(bar_length=fixed_length,
                     density=7850.0,
                     bulk_modulus=200e9,
                     number_of_elements=num_elems, 
                     horizon=fixed_horizon,
                     thickness=thickness)
                     #critical_stretch=1.0e-4)
    #problem1.introduce_flaw(0.0)
    #problem1.solve(max_time=1.0e-3, prescribed_force=1.0e8)
    problem1.solve(max_time=1.0e-3, prescribed_velocity=1.0)
    damage = problem1.compute_damage()

    mean_damage = damage.sum() / problem1.num_nodes
    mean_thickness = thickness.sum() / problem1.num_nodes
    max_thickness = thickness.max()

    return 0.5 * mean_damage + 0.5 * mean_thickness / max_thickness


### Main Program ####
if __name__ == "__main__":

    #Define problem size
    fixed_length = 10.0 
    delta_x = 0.25
    fixed_horizon = 2.6 * delta_x

    #Instantiate a 1d peridynamic problem with equally spaced nodes
    problem1 = PDJAX(bar_length=fixed_length,
                     density=7850.0,
                     bulk_modulus=200e9,
                     number_of_elements=int(fixed_length/delta_x), 
                     horizon=fixed_horizon,
                     thickness=0.25)
                     #critical_stretch=1.0e-4)
    #problem1.introduce_flaw(0.0)
    #problem1.solve(max_time=1.0e-3, prescribed_force=1.0e8)
    problem1.solve(max_time=1.0e-3, prescribed_velocity=1.0)

    fig, ax = plt.subplots()
    ax.plot(problem1.get_nodes(), problem1.get_solution(), 'k-')
    plt.show()

