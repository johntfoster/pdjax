#!/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import partial

import jax
import jax.numpy as jnp

import numpy as np
import scipy.spatial
import scipy.optimize
import matplotlib.pyplot as plt


import jax.scipy
import jax.scipy.optimize
from jax.scipy.optimize import minimize
from jax.nn import softplus

from jax import grad, jit


from typing import Union, Tuple

import optax
from jax import jit, vmap, grad, value_and_grad



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
                 critical_stretch:Union[float, None] = None,
                 prescribed_velocity:Union[float, None]=None, 
                 prescribed_force:Union[float, None]=None
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

        '''
        if isinstance(thickness, float):
            self.thickness = jnp.ones(number_of_elements) * thickness
        elif isinstance(thickness, np.ndarray):
            self.thickness = jnp.asarray(thickness)
            assert thickness.shape[0] == number_of_elements, ValueError("Thickness array length must match number of elements")
        '''
        if isinstance(thickness, float) or np.isscalar(thickness):
            self.thickness = jnp.ones(number_of_elements) * thickness
        elif isinstance(thickness, np.ndarray) or isinstance(thickness, jnp.ndarray):
            self.thickness = jnp.asarray(thickness)
            assert self.thickness.shape[0] == number_of_elements, ValueError("Thickness array length must match number of elements")
        else:
            raise ValueError("thickness must be a float or array")


        # Compute the pd_node locations, kdtree, nns, reference_position_state, etc.
        self.setup_discretization(self.thickness)

        # Debug plotting
        # _, ax = plt.subplots()
        # self.line, = ax.plot(self.pd_nodes, self.displacement)
        self._allow_damage = False

        self.critical_stretch = critical_stretch

        if self.critical_stretch is not None:
            self.allow_damage() 

        # Set boundary regions
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

        self.no_damage_region_left = jnp.asarray(self.tree.query_ball_point(self.pd_nodes[0, None], r=(2.5*self.horizon + np.max(self.lengths) / 2.0), p=2, eps=0.0)).sort()
        #:The node indices of the boundary region at the right end of the bar
        self.no_damage_region_right = jnp.asarray(self.tree.query_ball_point(self.pd_nodes[-1, None], r=(2.5*self.horizon + np.max(self.lengths) / 2.0), p=2, eps=0.0)).sort()


        # Compute the partial volumes
        vol_state, _ = self.compute_partial_volumes(self.thickness)

        # An  array containing the *influence vector-state* as defined in Silling et al. 2007
        # ratio = self.reference_magnitude_state / self.horizon
        # self.influence_state = jnp.ones_like(self.volume_state) - 35.0 * ratio ** 4.0 + 84.0 * ratio ** 5.0 - 70 * ratio ** 6.0 + 20 * ratio ** 7.0
        self.influence_state = jnp.where(vol_state > 1.0e-16, 1.0, 0.0)

        self.undamaged_influence_state = self.influence_state.copy()

        self.undamaged_influence_state_left = self.influence_state.at[self.no_damage_region_left, :].get()
        self.undamaged_influence_state_right = self.influence_state.at[self.no_damage_region_right, :].get()

        self.strain_energy_total = 0.0

        return


    def allow_damage(self):
        self._allow_damage = True
        return

    
    def setup_discretization(self, thickness:jax.Array):
        
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
        #set k=100, used to be k=6, will trim down later
        reference_magnitude_state, neighborhood = self.tree.query(self.pd_nodes[:,None], 
                k=100, p=2, eps=0.0, distance_upper_bound=(self.horizon + np.max(self.lengths) / 2.0))

        #jax.debug.print("ref_mag_state before: {r}",r=reference_magnitude_state)
        #trying to delete first column of ref_mag_state for broadcasting issue
        reference_magnitude_state = jnp.delete(reference_magnitude_state, 0, 1)  
        #jax.debug.print("ref_mag_state after initial trim: {r}",r=reference_magnitude_state )

        #jax.debug.print("ref_mag_state.shape after: {r}",r=reference_magnitude_state.shape)

        self.num_neighbors = jnp.asarray((neighborhood != self.tree.n).sum(axis=1)) - 1
        self.max_neighbors = np.max((neighborhood != self.tree.n).sum(axis=1))

        # Convert to JAX arrays and trim down excess neighbors
        neighborhood = jnp.asarray(neighborhood[:, :self.max_neighbors])
        #self.reference_magnitude_state = jnp.delete(reference_magnitude_state[:, :self.max_neighbors], 0,0)

        #changed to select just the first row and alleviate the broadcasting error
        #self.reference_magnitude_state = jnp.delete(reference_magnitude_state[1, :self.max_neighbors], 0,0)

        #jax.debug.print("neighborhood.shape: {n}",n=neighborhood.shape)
        #jax.debug.print("neighborhod: {n}",n=neighborhood)

        self.reference_magnitude_state = reference_magnitude_state[0:, :self.max_neighbors-1]
        #self.reference_magnitude_state = reference_magnitude_state[:, 1:self.max_neighbors]    
        

        #jax.debug.print("self.ref_mag_state.shape later: {r}",r=self.reference_magnitude_state)
        # Cleanup neighborhood
        row_indices = jnp.arange(neighborhood.shape[0]).reshape(-1, 1)
        neighborhood = jnp.where(neighborhood == self.tree.n, row_indices, neighborhood)
        self.neighborhood = jnp.delete(neighborhood,0,1)

        # Compute the reference_position_state.  Using the terminology of Silling et al. 2007
        self.reference_position_state = self.pd_nodes[self.neighborhood] - self.pd_nodes[:,None]

        # Cleanup reference_magnitude_state
        self.reference_magnitude_state = jnp.where(self.reference_magnitude_state == np.inf, 0.0, self.reference_magnitude_state)
        #jax.debug.print("self.ref_mag_state.shape at end of setup disc {r}",r=self.reference_magnitude_state.shape)

        return

    
    def compute_partial_volumes(self, thickness:jax.Array):

        # Setup some local (to function) convenience variables
        neigh = self.neighborhood
        lens = self.lengths
        ref_mag_state = self.reference_magnitude_state
        horiz = self.horizon

        #jax.debug.print("neigh: {v}", v=neigh.shape)

        # jax.debug.print("thickness in comp par v: {t}",t=thickness)
        # Initialize the volume_state to the lengths * width * thickness
        width = 1.0
        self.width = width
        vol_state_uncorrected = lens[neigh] * thickness[neigh] * width 

        #jax.debug.print("vol_state_unc in com p v: {v}", v=vol_state_uncorrected.shape)
        #jax.debug.print("ref_mag_state : {l}", l=ref_mag_state.shape)

        #Zero out entries that are not in the family
        vol_state_uncorrected = jnp.where(ref_mag_state < 1.0e-16, 0.0, vol_state_uncorrected) 
        
        # print(vol_state_uncorrected)


        #jax.debug.print("thickness: {t}", t=thickness)
        #jax.debug.print("neigh: {n}", n=neigh)
        #jax.debug.print("ref_mag_state: {r}", r=ref_mag_state)
        #jax.debug.print("vol_state_uncorrected: {v}", v=vol_state_uncorrected)
        #jax.debug.print("vol_state: {v}", v=vol_state)

        vol_state = jnp.where(ref_mag_state < horiz + lens[neigh] / 2.0, vol_state_uncorrected, 0.0)
        #print("vol_state:", vol_state)

        # Check to see if the neighboring node has a partial volume
        is_partial_volume = jnp.abs(horiz - ref_mag_state) < lens[neigh] / 2.0

        # Two different scenarios:
        is_partial_volume_case1 = is_partial_volume * (ref_mag_state >= horiz)
        is_partial_volume_case2 = is_partial_volume * (ref_mag_state < horiz)


        # Compute the partial volumes conditionally
        vol_state = jnp.where(is_partial_volume_case1, (lens[neigh] / 2.0 - (ref_mag_state - horiz)) * width * thickness[neigh], vol_state)
        vol_state = jnp.where(is_partial_volume_case2, (lens[neigh] / 2.0 + (horiz - ref_mag_state)) * width * thickness[neigh], vol_state)

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
        rev_vol_state = jnp.where(rev_vol_state > vol_array, vol_array, rev_vol_state)

        # Set attributes
        # self.volume_state = vol_state
        # self.reverse_volume_state = rev_vol_state

        return (vol_state, rev_vol_state)

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
                                vol_state:jax.Array,
                                inf_state:jax.Array) -> Tuple[jax.Array, jax.Array]:
         

        #Define some local convenience variables     
        ref_pos = self.pd_nodes 
        ref_pos_state = self.reference_position_state
        ref_mag_state = self.reference_magnitude_state
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
        if self._allow_damage :
            
            inf_state = jnp.where(stretch  > self.critical_stretch, 0.0, inf_state)
            # Disallow damage on the boundary regions

            inf_state = inf_state.at[self.no_damage_region_left, :].set(self.undamaged_influence_state_left)
            inf_state = inf_state.at[self.no_damage_region_right, :].set(self.undamaged_influence_state_right)
            # jax.debug.print('IS: {infl}', infl=inf_state.at[:].get())

        # Compute the shape tensor (really a scalar because this is 1d), i.e. the "weighted volume" as 
        # defined in Silling et al. 2007
        # added epsilon to prevent dividing by zero
        epsilon = 1e-12 
        shape_tens = (inf_state * ref_pos_state * ref_pos_state * vol_state).sum(axis=1)
        shape_tens = jnp.where(jnp.abs(shape_tens) < epsilon, epsilon, shape_tens)

        # Compute scalar force state for a elastic constitutive model
        ######### compute strain energy density here?  or calculation at least should look like this line here ########
        scalar_force_state = 9.0 * K / shape_tens[:, None] * exten_state
        # bond strain energy calc
        bond_strain_energy = 9.0 * K / shape_tens[:, None] * exten_state * exten_state * ref_mag_state

        #jax.debug.print("bond_strain_energy: {bse}", bse=bond_strain_energy)


        ###### calculating strain energy density here purely for debugging . . . actually do so in comp_int_forces###
        #strain_energy = (bond_strain_energy * vol_state).sum(axis=1)
        #total_strain_energy = jnp.sum(strain_energy)
        
        #strain_energy = (bond_strain_energy * vol_state)
        #jax.debug.print("total energy : {s}", s=total_strain_energy)
       
        # Compute the force state
        force_state = inf_state * scalar_force_state * def_unit_state

        # jax.debug.print('force_state: {f}', f=force_state.at[:].get())

        ###  return bond_strain_energy
        return force_state, inf_state, bond_strain_energy

    def compute_damage(self):
        return self._compute_damage(self.influence_state)
    '''
    def _compute_damage(self,vol_state:jax.Array, inf_state:jax.Array,  undamaged_inf_state:jax.Array):
        #jax.debug.print("inf_state in dam: {i}",i=inf_state)
        #jax.debug.print("inf_state.sum in dam: {i}",i=inf_state.sum(axis=1))   
        epsilon = 1e-12
        denom = (undamaged_inf_state * vol_state).sum(axis=1)
        denom = jnp.where(jnp.abs(denom) < epsilon, epsilon, denom)

        
        intDam = (inf_state * vol_state).sum(axis=1) / (undamaged_inf_state * vol_state).sum(axis=1)
        #intDam = (inf_state * vol_state).sum(axis=1) / (self.undamaged_influence_state * vol_state).sum(axis=1)
        # jax.debug.print(" dam / undam state sums: {i}",i=intDam)
        #jax.debug.print(" new dam shape: {i}",i=intDam.shape)

        #jax.debug.print(" new dam : {i}",i=intDam)

        #jax.debug.print("self.undamaged_infl_st: {s}",s=self.undamaged_influence_state)
        #jax.debug.print("vol_state: {v}",v=vol_state)
        #jax.debug.print("inf_state in dam: {i}",i=inf_state)

        intDamOG =inf_state.sum(axis=1) / self.num_neighbors
        #jax.debug.print(" OG dam: {i}",i=intDamOG)
        #jax.debug.print(" inf_state {i}",i=inf_state)

        

        #return 1 - ((inf_state * vol_state).sum(axis=1)) / ((undamaged_inf_state * vol_state).sum(axis=1))

        #return 1 - inf_state.sum(axis=1) / self.num_neighbors
        #return 1 - (inf_state * vol_state).sum(axis=1) / denom
        return 1 - intDamOG
        '''
    
    def _compute_damage(self, vol_state:jax.Array, inf_state:jax.Array, undamaged_inf_state:jax.Array):
        #return 1 - ((inf_state * vol_state).sum(axis=1)) / ((undamaged_inf_state * vol_state).sum(axis=1))
        #jax.debug.print("inf_state in dam: {i}",i=inf_state)
        #jax.debug.print("vol_state in dam: {i}",i=vol_state)
        #jax.debug.print("vol_state*inf_state: {i}",i=(inf_state * vol_state).sum(axis=1))
        #jax.debug.print("undamaged_inf_state in dam: {i}",i=undamaged_inf_state)
        #jax.debug.print("undamaged_inf_state*vol_state: {i}",i=(undamaged_inf_state * vol_state).sum(axis=1))
        #jax.debug.print("damaged/undamaged: {u}",u=(inf_state * vol_state).sum(axis=1)/(undamaged_inf_state * vol_state).sum(axis=1))

        #return (inf_state * vol_state).sum(axis=1) / self.num_neighbors
        return 1 - inf_state.sum(axis=1) / self.num_neighbors





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
    def compute_internal_force(self, disp, vol_state, rev_vol_state, inf_state, thickness, time):
            
        # Define some local convenience variables     
        neigh = self.neighborhood
        
        # Compute the force vector-state according to the choice of constitutive
        # model 
        # jax.debug.print("inf_state before c_f_st: {i}",i=inf_state) 
        
        ##### return bond_strain_energy #####
        force_state, inf_state, bond_strain_energy = self.compute_force_state_LPS(disp, vol_state, inf_state)
        # jax.debug.print("inf_state after c_f_st: {i}",i=inf_state) 


        #Integrate nodal forces 
        force = (force_state * vol_state).sum(axis=1)
        force = force.at[neigh].add(-force_state * rev_vol_state)

        strain_energy = (bond_strain_energy * vol_state).sum(axis=1)
        #total_strain_energy = jnp.sum(strain_energy)
        #jax.debug.print("strain_energy: {s}", s=jnp.sum(strain_energy))

        if self.prescribed_force is not None:
            li = 0
            ramp_force = self.smooth_ramp(time, t0=1.e-5, c=self.prescribed_force) 

            #trying to calculate force/unit area instead of normalized by volume
            #left_bc_force_density = ramp_force * (self.width * thickness[li]) / (vol_state[li].sum() + rev_vol_state[li][0])
            #left_bc_nodal_forces = left_bc_force_density * vol_state[li][self.left_bc_mask]
            #force = force.at[self.left_bc_region].add(-left_bc_nodal_forces)
            #force = force.at[li].add(-left_bc_force_density * rev_vol_state[li][li])

            #left_bc_area = self.width * thickness[self.left_bc_region]
            #left_bc_nodal_forces = ramp_force * left_bc_area
            left_bc_nodal_forces = (ramp_force * self.width * self.width)/(vol_state[li].sum() + rev_vol_state[li][0])
            force = force.at[self.left_bc_region].add(-left_bc_nodal_forces)
            # For the leftmost node (if needed)
            #left_bc_area_li = self.width * thickness[li]
            #force = force.at[li].add(-ramp_force * left_bc_area_li)
            #
            ri = self.num_nodes - 1
            #right_bc_force_density = ramp_force * (self.width) / (vol_state[ri].sum() + rev_vol_state[ri][0])
            #right_bc_force_density = ramp_force * (self.width * thickness[ri]) / (vol_state[ri].sum() + rev_vol_state[ri][0])
            #right_bc_nodal_forces = right_bc_force_density * vol_state[ri][self.right_bc_mask]
            #force = force.at[self.right_bc_region].add(right_bc_nodal_forces)
            #force = force.at[ri].add(right_bc_force_density * rev_vol_state[ri][0])

            #trying to calculate force/unit area instead of normalized by volume
            #right_bc_area = self.width * thickness[self.right_bc_region]
            #right_bc_nodal_forces = ramp_force * right_bc_area
            right_bc_nodal_forces = (ramp_force * self.width * self.width)/(vol_state[li].sum() + rev_vol_state[li][0])
            force = force.at[self.right_bc_region].add(right_bc_nodal_forces)
            # For the rightmost node (if needed)
            #right_bc_area_ri = self.width * thickness[ri]
            #force = force.at[ri].add(ramp_force * right_bc_area_ri)

            #jax.debug.print("force at left bc region: {f}", f=force.at[self.left_bc_region].get())
            #jax.debug.print("force at right bc region: {f}", f=force.at[self.right_bc_region].get())
            #jax.debug.print("strain_energy: {s}", s=jnp.sum(strain_energy))
        return force, inf_state, strain_energy


    def solve_one_step(self, vals:Tuple[jax.Array, jax.Array, jax.Array, 
                                        jax.Array, jax.Array, jax.Array, 
                                        jax.Array, jax.Array, jax.Array, float, float]):

        (disp, vel, acc, vol_state, rev_vol_state, inf_state, thickness, undamaged_inf_state, strain_energy_total, time) = vals

        # TODO: Solve for stable time step
        time_step = 1.0e-6


        if self.prescribed_velocity is not None:
            bc_value = self.prescribed_velocity * time
            # Apply displacements bcs
            f = lambda x: 2.0 * bc_value / self.bar_length * x
            disp = disp.at[self.left_bc_region].set(f(self.pd_nodes[self.left_bc_region]))
            disp = disp.at[self.right_bc_region].set(f(self.pd_nodes[self.right_bc_region]))

        #### return strain_energy here #####
        force, inf_state, strain_energy = self.compute_internal_force(disp, vol_state, rev_vol_state, inf_state, thickness, time)

        acc_new = force / self.rho
        vel = vel.at[:].add(0.5 * (acc + acc_new) * time_step)
        disp = disp.at[:].add(vel * time_step + (0.5 * acc_new * time_step * time_step))
        acc = acc.at[:].set(acc_new)

        

        strain_energy_total = jnp.sum(strain_energy)
        #jax.debug.print("strain_energy_total in solve_one_step: {tse}", tse=strain_energy_total)

        #return (disp, vel, acc, vol_state, rev_vol_state, inf_state, thickness, undamaged_inf_state, time + time_step)
        return (disp, vel, acc, vol_state, rev_vol_state, inf_state, thickness, undamaged_inf_state, strain_energy_total, time + time_step)



    def _solve(self, 
              thickness:Union[jax.Array, None]=None,
              max_time:float=1.0):
        '''
            Solves in time using Verlet-Velocity
        '''
        #jax.debug.print("max_time: {t}",t=max_time)
        time_step = 1.0e-6
        #time_step = 1.0e-6
        num_steps = int(max_time / time_step)

        if thickness is None:
            thickness = self.thickness

        vol_state, rev_vol_state = self.compute_partial_volumes(thickness)


        #jax.debug.print("vol_state aft comp p v: {vs}",vs=vol_state)
        #jax.debug.print("rev_vol_s update: {rv}",rv=rev_vol_state)

        inf_state = self.influence_state.copy() 
        undamaged_inf_state = self.undamaged_influence_state.copy()
        # Initialize a fresh influence state for this run
        #inf_state = jnp.where(vol_state > 1.0e-16, 1.0, 0.0)

        #jax.debug.print("inf_state update after where: {i}",i=inf_state)
        # The fields
        disp = jnp.zeros_like(self.pd_nodes)
        vel = jnp.zeros_like(self.pd_nodes)
        acc = jnp.zeros_like(self.pd_nodes)
        time = 0.0
        strain_energy_init = self.strain_energy_total
        #strain_energy = jnp.ones(self.number_of_elements)
        #strain_energy_init = self.strain_energy_total
        #strain_energy_init = 0.0 

    

        # Define loop body (takes index and current vals)
        def loop_body(i, vals):
            jax.debug.print("entering solve_one_step: {i}", i=i)
            vals = self.solve_one_step(vals)
            return vals

        #Solve
        vals = (disp, vel, acc, vol_state, rev_vol_state, inf_state, thickness, undamaged_inf_state, strain_energy_init, time)


        #vals = jax.lax.fori_loop(0, num_steps, loop_body, vals)

        #### added this section in place of while loop
        ##### when implemented new more detailed damage funct it did not like while loop
        
        def scan_step(vals, _):
            return self.solve_one_step(vals), None

        vals, _ = jax.lax.scan(scan_step, vals, None, length=num_steps)

        #jax.debug.print("strain_energy_total in solve: {tse}", tse=jnp.sum(vals[8]))
        #vals = jax.lax.while_loop(lambda vals: vals[8] < max_time, self.solve_one_step, vals)
        #self.volume_state = vals[3]
        

        '''
        #added the following lines to calculate the vals within solve_one_step in place of while loop above
        num_steps = int(max_time / time_step)

        def scan_body(carry, _):
            return self.solve_one_step(carry), None
        
        #jax.debug.print("inf_state in solve bef {}", inf_state)

        init_vals = (disp, vel, acc, vol_state, rev_vol_state, inf_state, thickness, undamaged_inf_state, strain_energy_total, time)
        final_vals, _ = jax.lax.scan(scan_body, init_vals, None, length=num_steps)


        '''
        jax.debug.print("final strain_energy_total in solve: {tse}", tse=jnp.sum(vals[8]))
        jax.debug.print("thickness in solve: {i}", i=vals[6])

        
        return vals


    def solve(self, max_time:float=1.0):
        '''
            Solves in time using Verlet-Velocity
        '''

        vals = self._solve(max_time=max_time)

        self.displacement = vals[0]
        self.velocity = vals[1]
        self.acceleration = vals[2]
        self.influence_state = vals[5]
        self.strain_energy_total = vals[8] 


        return 


    def get_solution(self):
        ''' Convenience function for retrieving displacement'''
        return self.displacement

    def get_nodes(self):
        ''' Convenience function for retrieving peridynamic node locations'''
        return self.pd_nodes
    
#def loss(thickness:jax.Array, problem:PDJAX, max_time=1.0e-3):
def loss(thickness:jax.Array, problem:PDJAX, max_time=1.0E-8):
    
    ###################################################
    #jax.debug.print("thickness in loss: {th}", th=thickness)

    min_thickness = 1e-3  # or something physically reasonable
    #thickness = jnp.clip(thickness, a_min=min_thickness) 
    thickness = softplus(thickness)
    #jax.debug.print("thickness in loss: {th}", th=thickness)

    vals = problem._solve(thickness, max_time=max_time)
    
    #### set loss value to strain_energy #######
    ### using total strain energy of bar with thickness 1.0
    normalization_factor = 50.0
    total_strain_energy = jnp.sum(vals[8])
    #loss_value = total_strain_energy / normalizaion_factor
    max_thickness = jnp.max(thickness)
    #loss_value = max_thickness

    loss_value = 0.9 * (total_strain_energy/normalization_factor) + 0.1 * max_thickness

    
    jax.debug.print("max thickness: {mt}", mt=max_thickness)
    jax.debug.print("total strain energy: {tse}", tse=total_strain_energy/normalization_factor)
    jax.debug.print("loss: {l}", l=loss_value)
    #####################################################################


    return loss_value



### Main Program ####
if __name__ == "__main__":

    #Define problem size
    fixed_length = 10.0
    delta_x = 0.25
    fixed_horizon = 3.6 * delta_x
    #increased horizon to 3.6 from 2.6, smoothes things out and makes them more stable
    #error might have started then?

    # Initial parameter (scalar for thickness)
    key = jax.random.PRNGKey(0)  # Seed for reproducibility

    # Create a random array with values between 0.5 and 1.0
    shape = (fixed_length/delta_x,)  # Example shape (adjust as needed)
    #minval = 1.15
    minval = 0.5
    maxval = 1.15
    #thickness = jax.random.uniform(key, shape=shape, minval=minval, maxval=maxval)
    thickness = 1.5

    #Instantiate a 1d peridynamic problem with equally spaced nodes
    problem1 = PDJAX(bar_length=fixed_length,
                     density=7850.0,
                     bulk_modulus=200e9,
                     number_of_elements=int(fixed_length/delta_x), 
                     horizon=fixed_horizon,
                     thickness=thickness,
                     prescribed_force=1.0e8,
                     critical_stretch=1.0e-4)

    ######### to run forward problem ###############
    problem1.solve(max_time=1.0e-3)
    

    
    fig, ax = plt.subplots()
    ax.plot(problem1.get_nodes(), problem1.get_solution(), 'ko')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'displacement')
    plt.show()

    ################################################

    ############### to run optax optimization ###########
        
    ################  now using optax to maximize ##########################
    # Initial parameter (scalar for thickness)
    key = jax.random.PRNGKey(0)  # Seed for reproducibility

    # Create a random array with values between 0.5 and 1.0
    shape = (problem1.num_nodes,)  # Example shape (adjust as needed)
    #minval = 1.15
    minval = 0.5
    maxval = 1.15
    param = jax.random.uniform(key, shape=shape, minval=minval, maxval=maxval)
    #param =  jnp.ones(problem1.num_nodes) * # Initial thickness guess
    
    #param = thickness
    #param = jnp.array([2.0])
    learning_rate = 1
    num_steps = 4



    # Optax optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(param)

    # Loss function (already defined as 'loss')
    loss_and_grad = jax.value_and_grad(loss)

    # Optimization loop
    for step in range(num_steps):
        loss_val, grads = loss_and_grad(param, problem1)

        print(f"Step {step},  loss: {loss_val}, grads: {grads}")

        if jnp.isnan(loss_val) or jnp.any(jnp.isnan(grads)):
            print("NaN detected! Stopping optimization.")
            break

        updates, opt_state = optimizer.update(grads, opt_state, param)
        param = optax.apply_updates(param, updates)
        #print("updated param: ", param)
        if step % 20 == 0:
            print(f"Step {step}, loss: {loss_val}")
            #print(f"Step {step}, loss: {loss_val}, param: {param}")

    # Use the optimized thickness
    opt_thickness = softplus(param)
    #vals = problem1._solve(opt_thickness)
    print("opt_thickness: ", opt_thickness)
    '''
    print("disp: ", vals[0])
    fig, ax = plt.subplots()
    ax.plot(problem1.get_nodes(), vals[0], 'ko')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'displacement')
    plt.show()
    '''

    ########################################################

    ###############  scipy optimization ###########
  
    '''
    #######################################
    #problem1.introduce_flaw(0.0)

    #thickness = jnp.ones(problem1.num_nodes) * 0.5
    #problem1.solve(max_time=1.0e-3)
    # #
    #print(vals[0])

    #fig, ax = plt.subplots()
    #ax.plot(problem1.get_nodes(), problem1.get_solution(), 'ko')
    #ax.set_xlabel(r'$x$')
    #ax.set_ylabel(r'displacement')
    #plt.show()
    key = jax.random.PRNGKey(0)  # Seed for reproducibility

    # Create a random array with values between 0.5 and 1.0
    #shape = (problem1.num_nodes,)  # Example shape (adjust as needed)
    minval = 0.5
    maxval = 1.0
    thickness = jax.random.uniform(key, shape=shape, minval=minval, maxval=maxval)
    result = jax.scipy.optimize.minimize(loss, thickness, args=(problem1,), method='BFGS', tol=0.1)
    #print("opt result.x,thickness: ",softplus(result.x))


    #thickness = jax.random.uniform(key, shape=shape, minval=minval, maxval=maxval)
    #scalar_int = jnp.array([2.0])  # Make it a 1-element array
    #thickness = jnp.ones(problem1.number_of_elements)
    #thickness_scaled = 2.0 * thickness

    #problem1.solve(max_time=1.0e-3)


    #init_thick = softplus(scalar_int[0]) * jnp.ones(problem1.number_of_elements)
    #pt_scalar = softplus(result.x)
    #opt_thickness = opt_scalar * jnp.ones(problem1.number_of_elements)
 
    #result = jax.scipy.optimize.minimize(loss, scalar_int, args=(problem1,), method='BFGS') 


    #init_thick = softplus(scalar_int[0]) * jnp.ones(problem1.number_of_elements)
    opt_thickness = softplus(result.x)
    #opt_thickness_no_softplus = result.x
    #opt_thickness = opt_scalar * jnp.ones(problem1.number_of_elements)




    #opt_thickness = pt_scalar
    #vals = problem1._solve(opt_thickness)
    print("opt_thickness: ",opt_thickness)
    #print("opt_thickness_no_softplus: ",opt_thickness_no_softplus)
    print("disp: ",vals[0])
    fig, ax = plt.subplots()
    ax.plot(problem1.get_nodes(), vals[0], 'ko')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'displacement')
    plt.show()
    ###################################################
    '''

