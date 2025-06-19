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
#from jaxopt import ScipyMinimize

from jax import grad, jit

#from jax.scipy.optimize import minimize

#import jax
#import jax.numpy as jnp
#from jax import grad, jit
#import numpy as np

from typing import Union, Tuple

import matplotlib.pyplot as plt


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
                 prescribed_velocity:Union[float, None] = None,
                 prescribed_traction:Union[float, None] = None
                 ) -> None:
        '''
           Initialization function
        '''

        #Problem data
        self.bulk_modulus = bulk_modulus
        self.rho = density
        

        self.bar_length = bar_length
        self.number_of_elements = number_of_elements

        self.prescribed_velocity = prescribed_velocity
        self.prescribed_traction = prescribed_traction

        width = 1.0
        self.width = width

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


        #thickness[20] = -0.01
        #thickness=jnp.asarray(thickness)

        
        # Compute the pd_node locations, kdtree, nns, reference_position_state, etc.
        self.setup_discretization(thickness,width)

        # Debug plotting
        # _, ax = plt.subplots()
        # self.line, = ax.plot(self.pd_nodes, self.displacement)
        self._allow_damage = True

        self.critical_stretch = critical_stretch

        if self.critical_stretch is not None:
            self.allow_damage() 
        

        return

    def reset(self, displacement, velocity, acceleration, influence_state, thickness, width):
        '''
            Resets the problem to the initial state, i.e. zero displacements, velocities, etc.
        '''

        displacement = displacement.at[:].set(0.0)
        velocity = velocity.at[:].set(0.0)
        acceleration = acceleration.at[:].set(0.0)
        influence_state = jnp.where(..., 1.0, 0.0)

        #isplacement = jnp.zeros_like(self.displacement)
        #velocity = jnp.zeros_like(self.velocity)
        #acceleration = jnp.zeros_like(self.acceleration)
        #influence_state = jnp.where(self.volume_state > 1.0e-16, 1.0, 0.0)
        # Compute partial volumes as needed, return as well if needed

        #self.displacement = self.displacement.at[:].set(0.0)
        #self.velocity = self.velocity.at[:].set(0.0)
        #self.acceleration = self.acceleration.at[:].set(0.0)
        #self.influence_state = jnp.where(self.volume_state > 1.0e-16, 1.0, 0.0)

        vol_state,rev_vol_state=self.compute_partial_volumes(width, thickness)

        return displacement, velocity, acceleration, influence_state



    def allow_damage(self):
        self._allow_damage = True
        return


    
    def setup_discretization(self, thickness:np.ndarray, width):
        
        nodes = self.nodes
        prescribed_velocity = self.prescribed_velocity
        #print("prescribed_velocity: ", prescribed_velocity)

        # The lengths of the *elements*
        self.lengths = jnp.array(nodes[1:] - nodes[0:-1]) 

        # The PD nodes are the centroids of the elements
        self.pd_nodes = jnp.array(nodes[0:-1] + self.lengths / 2.0)
        self.num_nodes = self.pd_nodes.shape[0]

        # Creates a kdtree to do nearest neighbor search
        # 2D array is created from this search 

        self.tree = scipy.spatial.cKDTree(self.pd_nodes[:,None])
        #print(self.tree.data.shape)

        # Get PD nodes in the neighborhood of support + largest node spacing, this will
        # find all potential partial volume nodes as well. The distances returned from the
        # search turn out to be the reference_magnitude_state, so we'll store them now
        # to avoid needed to calculate later.
        reference_magnitude_state, neighborhood = self.tree.query(self.pd_nodes[:,None], 
                k=100, p=2, eps=0.0, distance_upper_bound=(self.horizon + np.max(self.lengths) / 2.0))
     
        #trying to delete first column of ref_mag_state for broadcasting issue
        #reference_magnitude_state = jnp.delete(reference_magnitude_state, 0, 1)  

        self.num_neighbors = jnp.asarray((neighborhood != self.tree.n).sum(axis=1)) - 1
        self.max_neighbors = jnp.max((neighborhood != self.tree.n).sum(axis=1))

        # Convert to JAX arrays and trim down excess neighbors
        neighborhood = jnp.asarray(neighborhood[:, :self.max_neighbors])
        self.reference_magnitude_state = jnp.delete(reference_magnitude_state[:, :self.max_neighbors], 0,0)

        #changed to select just the first row and alleviate the broadcasting error
        #self.reference_magnitude_state = jnp.delete(reference_magnitude_state[:, :self.max_neighbors], 0,0)
        #switching to slicing instead of jnp.delete
        # want to be able to delete the first column of reference_magnitude_state, 1: resolves broadcasting error here
        self.reference_magnitude_state = reference_magnitude_state[0:, 1:self.max_neighbors]
      


        # Cleanup neighborhood
        row_indices = jnp.arange(neighborhood.shape[0]).reshape(-1, 1)
        neighborhood = jnp.where(neighborhood == self.tree.n, row_indices, neighborhood)
        ############## feel like this line was meant to delete the first column since that is the node itself? ##########
        self.neighborhood = jnp.delete(neighborhood,0,1)


        # Compute the reference_position_state.  Using the terminology of Silling et al. 2007
        self.reference_position_state = self.pd_nodes[self.neighborhood] - self.pd_nodes[:,None]
        #print("self.ref_pos_st: ",self.reference_position_state)

        # Compute the partial volumes
        vol_state, rev_vol_state = self.compute_partial_volumes(width, thickness)

        # Cleanup reference_magnitude_state
        self.reference_magnitude_state = jnp.where(self.reference_magnitude_state == np.inf, 0.0, self.reference_magnitude_state)

        # A Numpy masked array containing the *influence vector-state* as defined in Silling et al. 2007
        # ratio = self.reference_magnitude_state / self.horizon
        # self.influence_state = jnp.ones_like(self.volume_state) - 35.0 * ratio ** 4.0 + 84.0 * ratio ** 5.0 - 70 * ratio ** 6.0 + 20 * ratio ** 7.0
        self.influence_state = jnp.where(vol_state > 1.0e-16, 1.0, 0.0)
        #print("self.influence_state in setup disc: ", self.influence_state)

        #:The node indices of the boundary region at the left end of the bar
        li = 0
        self.left_bc_mask = self.neighborhood[li] != li
        disp = self.neighborhood[li][self.left_bc_mask]


        ri = self.num_nodes - 1
        self.right_bc_mask = self.neighborhood[ri] != ri
        self.right_bc_region = self.neighborhood[ri][self.right_bc_mask]

        #the left and right boundary regions are the nodes that are within the horizon distance
        #self.prescribed_velocity = prescribed_velocity
        #print("self.prescribed_velocity: ", self.prescribed_velocity)

        pd_node0 = float(self.pd_nodes[0])
        pd_nodeN = float(self.pd_nodes[-1])

        if prescribed_velocity is not None:
            self.left_bc_region = np.array(self.tree.query_ball_point([pd_node0], r=(self.horizon + np.max(self.lengths) / 2.0), p=2, eps=0.0))
            self.right_bc_region = np.array(self.tree.query_ball_point([pd_nodeN], r=(self.horizon + np.max(self.lengths) / 2.0), p=2, eps=0.0))
        


        self.no_damage_region_left = np.array(self.tree.query_ball_point([pd_node0], r=(2.0*self.horizon + np.max(self.lengths) / 2.0), p=2, eps=0.0))
        self.no_damage_region_right = np.array(self.tree.query_ball_point([pd_nodeN], r=(2.0*self.horizon + np.max(self.lengths) / 2.0), p=2, eps=0.0))

        return


    def intermed_setup_disc(self, width, thickness):
        #jax.debug.print("thickness in intermed set disc: {}",thickness)
        # implementing a lower bound for thickness
        thickness = jnp.clip(thickness, 1e-3, None)  # Prevent zero/negative thickness
        #problem1.thickness = thickness  # Update the thickness in the problem object

        # Compute the partial volumes
        vol_state,rev_vol_state=self.compute_partial_volumes(width, thickness)

        # Compute inf state from those volumes
        inf_state = jnp.where(vol_state > 1.0e-16, 1.0, 0.0)

        return(vol_state, rev_vol_state, inf_state)

    
    def compute_partial_volumes(self, width, thickness:np.ndarray):

        # Setup some local (to function) convenience variables
        neigh = self.neighborhood
        lens = self.lengths
        ref_mag_state = self.reference_magnitude_state
        horiz = self.horizon

        # implementing a lower bound for thickness
        # thickness = jnp.clip(thickness, 1e-3, None)  # Prevent zero/negative thickness
        #problem1.thickness = thickness  # Update the thickness in the problem object  

        #jax.debug.print("t in comp part vol: {}",thickness)

        #print("neigh = ",neigh)
        #print("shapes:",neigh.shape,ref_mag_state.shape,vol_state.shape)

        # Initialize the volume_state to the lengths * width * thickness
        #width = 1.0
        #self.width = width
        #self.thickness = thickness
        vol_state_uncorrected = lens[neigh] * thickness[neigh] * width 
        horizArray = jnp.ones(ref_mag_state.shape) * horiz 

        #may have to change this back to horiz from horiz array
        #print("horiz array shape ",horizArray.shape)
        #print("ref-horizAr shapes: ", ref_mag_state.shape,neigh.shape,vol_state_uncorrected.shape)    
        #should be all good here for broadcasting dim

        #print("lens[neigh]: ",lens[neigh])
        vol_state = jnp.where(ref_mag_state < horizArray + lens[neigh] / 2.0, vol_state_uncorrected, 0.0)

        # Check to see if the neighboring node has a partial volume
        is_partial_volume = jnp.abs(horizArray - ref_mag_state) < lens[neigh] / 2.0

        #print("is_partial_volume= ",is_partial_volume)
        #print("shape is_partial= ",is_partial_volume.shape)

        # Two different scenarios:
        is_partial_volume_case1 = is_partial_volume * (ref_mag_state >= horizArray)
        is_partial_volume_case2 = is_partial_volume * (ref_mag_state < horizArray)

        neighT=np.transpose(neigh) 

        #print("shapes:",neighT.shape)
        
        #making horiz into an array, was previously a scalar

        # Compute the partial volumes conditionally
        vol_state = jnp.where(is_partial_volume_case1, (lens[neigh] / 2.0 - (ref_mag_state - horizArray)) * width * thickness[neigh], vol_state)

        vol_state = jnp.where(is_partial_volume_case2, (lens[neigh] / 2.0 + (horizArray - ref_mag_state)) * width * thickness[neigh], vol_state)

        #print("(ref-horiz)*w*t= ",(lens[neigh] / 2.0 - (ref_mag_state - horizArray))*width*thickness[neigh])
        #print("shape ref-horiz= ",(lens[neigh] / 2.0 - (ref_mag_state - horizArray)).shape)

        #print("vol_state_uncorrected: ",vol_state_uncorrected)


        # If the partial volume is predicted to be larger than the unocrrected volume, set it back
        # vol_state = jnp.where(vol_state > vol_state_uncorrected, vol_state_uncorrected, vol_state)
        vol_state = jnp.where(vol_state > vol_state_uncorrected, vol_state_uncorrected, vol_state)

        #print("ref-horizAr: ", ref_mag_state-horizArray) 

        # Now compute the "reverse volume state", this is the partial volume of the "source" node, i.e. node i,
        # as seen from node j.  This doesn't show up anywhere in any papers, it's just used here for computational
        # convenience
        vol_array = lens[:,None] * width * thickness[:, None]

        #print("vol_array: ",vol_array)
        #print("vol_array.shape: ",vol_array.shape)

        rev_vol_state = jnp.ones_like(vol_state) * vol_array
        rev_vol_state = jnp.where(is_partial_volume_case1, (lens[:, None] / 2.0 - (ref_mag_state - horizArray)) * width * thickness[:, None], rev_vol_state)
        rev_vol_state = jnp.where(is_partial_volume_case2, (lens[:, None] / 2.0 + (horizArray - ref_mag_state)) * width * thickness[:, None], rev_vol_state)
        #print("rev_vol_state= ",rev_vol_state)
        #If the partial volume is predicted to be larger than the uncorrected volume, set it back
        rev_vol_state = jnp.where(rev_vol_state > vol_array, vol_array, rev_vol_state)
        #print("vol_state in comput  part vo= ",vol_state)



        # Set attributes
        #self.volume_state = vol_state
        #self.reverse_volume_state = rev_vol_state

        return (vol_state, rev_vol_state)

    def introduce_flaw(self, location:float, allow_damage=False):  
        if allow_damage:
            self.allow_damage()

        #converting ot a numpy array of integers for use in scipy.spatial.cKDTree.query
        max_neighbors = int(jax.device_get(self.max_neighbors))

        _, nodes_near_flaw = self.tree.query(location, k=max_neighbors, p=2, eps=0.0, 
                                             distance_upper_bound=(self.horizon + np.max(self.lengths)/2))

        # The search above will produce duplicate neighbor nodes, make them into a
        # unique 1-dimensional list
        nodes_near_flaw = np.array(np.unique(nodes_near_flaw), dtype=np.int64)
        #print("nodes_near_flaw: ",nodes_near_flaw)
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
                                inf_state:jax.Array,
                                vol_state, 
                                rev_vol_state) -> Tuple[jax.Array, jax.Array]:
         
        #print("inf_state.shape: ",inf_state.shape)
        #print("inf_state: ",inf_state)
        #Define some local convenience variables     
        ref_pos = self.pd_nodes 
        ref_pos_state = self.reference_position_state
        ref_mag_state = self.reference_magnitude_state
        #vol_state = self.volume_state
        neigh = self.neighborhood
        K = self.bulk_modulus
        #inf_state = self.influence_state
        #changed inf_state def above here because was being created as a single row vecor, now it is a 2D array with shape (num_nodes, num_neighbors)

        #Compute the deformed positions of the nodes
        def_pos = ref_pos + disp
        #print("def_pos: ",def_pos)
        #print("def_pos.shape :",def_pos.shape)

        #Compute deformation state
        def_state = def_pos[neigh] - def_pos[:,None]
        #print("def_state: ",def_state)
        #print("def_state.shape :",def_state.shape)

        # Compute deformation magnitude state
        def_mag_state = jnp.sqrt(def_state * def_state)
        #print("def_mag_state: ",def_mag_state)
        #print("def_mag_state.shape :",def_mag_state.shape)

        # Compute deformation unit state
        def_unit_state = jnp.where(def_mag_state > 1.0e-16, def_state / def_mag_state, 0.0)
   
        # Compute scalar extension state
        exten_state = def_mag_state - ref_mag_state
        #print("exten_state: ",exten_state.shape)    
        #jax.debug.print("exten/def_mag: ", exten_state/ref_mag_state)


        stretch = jnp.where(ref_mag_state > 1.0e-16, exten_state / ref_mag_state, 0.0)
        
        # Apply a critical strech fracture criteria
        print("damage= ",self._allow_damage)
        #jax.debug.print("stretch:  {}",stretch)

        def apply_damage(inf_state):
            # Apply damage where stretch exceeds critical_stretch
            inf_state = jnp.where(stretch > self.critical_stretch, 0.0, inf_state)
            # Disallow damage on the boundary regions
            undamaged_influence_state_left = self.influence_state.at[self.no_damage_region_left, :].get()
            inf_state = inf_state.at[self.no_damage_region_left, :].set(undamaged_influence_state_left)
            undamaged_influence_state_right = self.influence_state.at[self.no_damage_region_right, :].get()
            inf_state = inf_state.at[self.no_damage_region_right, :].set(undamaged_influence_state_right)
            return inf_state

        inf_state = jax.lax.cond(self._allow_damage, apply_damage,lambda x: x,inf_state)

        #replaced below if statement with a funtion call that uses jax.lax to avoid nonconcrete boolean errors
        '''
        if self._allow_damage == True:
            #print("stretch shape: ",stretch.shape)
            #print("inf_state: ",inf_state)
            #print("inf_state.shape: ",inf_state.shape)

            ############### this is where the broadcasting error occurs ##################
            inf_state = jnp.where(stretch  > self.critical_stretch, 0.0, inf_state)
            #print("inf_state: ",inf_state)
            #print("inf_state.shape: ",inf_state.shape)
            

            # Disallow damage on the boundary regions
            undamaged_influence_state_left = self.influence_state.at[self.no_damage_region_left, :].get()
            inf_state = inf_state.at[self.no_damage_region_left, :].set(undamaged_influence_state_left)
            undamaged_influence_state_right = self.influence_state.at[self.no_damage_region_right, :].get()
            inf_state = inf_state.at[self.no_damage_region_right, :].set(undamaged_influence_state_right)
            #jax.debug.print('IS: {infl}', infl=inf_state.at[:].get())
            #print("volState.shape :",vol_state.shape)
        '''

        # Compute the shape tensor (really a scalar because this is 1d), i.e. the "weighted volume" as 
        # defined in Silling et al. 2007
        #shape_tens = (inf_state * ref_pos_state * ref_pos_state * vol_state)
        #shape_tens = (inf_state * ref_pos_state * vol_state).sum(axis=1)
        #print("shape_tens= ",shape_tens)
        shape_tens = jax.device_get((inf_state * ref_pos_state * ref_pos_state * vol_state).sum(axis=1))
        
        #print("exten_state.shape: ",exten_state.shape)
        #print("k/shape_tens: ",9.0* K / shape_tens[:, None])


        # Compute scalar force state for a elastic constitutive model
        scalar_force_state = 9.0 * K / shape_tens[:, None] * exten_state
        #print("scalar_force_s.shape: ",scalar_force_state.shape)
        #print(inf_state.shape,scalar_force_state.shape,def_unit_state.shape)

        # Compute the force state
        force_state = inf_state * scalar_force_state * def_unit_state

        #print("ForceSt.shape,inf_state.shape: ",force_state.shape,inf_state.shape)

        return force_state, inf_state

    def compute_damage(self,inf_state):

        #var_influence_state = self.influence_state.sum(axis=1)
        #jax.debug.print("inf_state in comp dam: {}", inf_state)
        var_inf_state = inf_state.sum(axis=1)
        #var=self.influence_state.sum(axis=1)
        #jax.debug.print("var_inf_state: {}",var_inf_state)
        #jax.debug.print("self.num_neighbors: {}",self.num_neighbors)      

        #print("mean_damage: ",jax.debug.print("mean_damage: {}", mean_damage))
        #print("var.shape: ",var.shape)
        #print("num_neighbors: ",self.num_neighbors)
        #print("num_neigh shape ",self.num_neighbors.shape)
        int_damage = jnp.divide(var_inf_state,self.num_neighbors)
        #jax.debug.print("int_damage: {}",int_damage)

        damage = 1 - var_inf_state / self.num_neighbors
        #print("damage in comp dam:",damage)

        return damage


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
    def compute_internal_force(self, disp, inf_state, time, vol_state, rev_vol_state, thickness, width):
            
        # Define some local convenience variables     
        #vol_state = self.volume_state
        #rev_vol_state = self.reverse_volume_state
        neigh = self.neighborhood
        #print("inf_state.shape: ",inf_state.shape)
        #print("inf_state: ",inf_state)

        # implementing a lower bound for thickness
        thickness = jnp.clip(thickness, 1e-3, None)  # Prevent zero/negative thickness
        #problem1.thickness = thickness  # Update the thickness in the problem object

        #jax.debug.print("t begin comp int for: {}",thickness)
        
        # Compute the force vector-state according to the choice of constitutive
        # model  
        #jax.debug.print("BEFORE COMPUTE FORCE STATE F {}", inf_state)
        force_state, inf_state = self.compute_force_state_LPS(disp, inf_state, vol_state, rev_vol_state)


        #print("force_st,inf_state= ",force_state,inf_state)

        #Integrate nodal forces 
        force = (force_state * vol_state).sum(axis=1)

        force = force.at[neigh].add(-force_state * rev_vol_state)
        
        #indices_l_bc_mask = np.array(jnp.where(self.left_bc_mask)[0])

        if self.prescribed_traction is not None:
            li = 0

            # Compute the left and right boundary conditions
            # Changed syntax to alleviate broadcasting error
            # Also were indexing using the jax array indicies which caused an error so assigend more local variables
            ramp_traction = self.smooth_ramp(time, t0=1.e-5, c=self.prescribed_traction)  

            denom = vol_state[li].sum() + rev_vol_state[li][0]
            denom = jnp.where(denom < 1e-12, 1e-12, denom)
            left_bc_force_density = ramp_traction * (self.width * thickness[li]) / denom

            #left_bc_force_density = ramp_traction * (width * thickness[li]) / (vol_state[li].sum() + rev_vol_state[li][0])

            mask = self.left_bc_mask
            #print("mask.shape: ",mask.shape)
            left_bc_nodal_forces = left_bc_force_density * vol_state[li] * mask
            force = force.at[li].add(-left_bc_nodal_forces.sum())




            ## think the below line here is causing the broadcasting error, need to check shape of region
            #force = force.at[self.left_bc_region].add(-left_bc_nodal_forces)

            #print("left_BC_region: ", self.left_bc_region)

            # left_bc_nodal_forces is full length, so index it
            #forces_to_add = left_bc_nodal_forces[self.left_bc_region]
            # Add only to the specified indices
            #force = force.at[self.left_bc_region].add(-forces_to_add)
            #print("force after left bc region: ", jax.device_get(force.shape))

            #force = force.at[li].add(-left_bc_force_density * self.reverse_volume_state[li][li])
            #force = force - left_bc_nodal_forces * self.left_bc_mask
            #print("force after left  with density: ", jax.device_get(force.shape))
            

            li = 0
            neighbor_indices = self.neighborhood[li]  # shape (num_neighbors,)
            mask = self.left_bc_mask                  # shape (num_neighbors,)
            left_bc_nodal_forces = left_bc_force_density * vol_state[li] * mask  # shape (num_neighbors,)

            # Only update the neighbors where mask is True
            force = force.at[neighbor_indices].add(-left_bc_nodal_forces)


            ri = self.num_nodes - 1
            #implementing check to be sure denom isn't zero 
            denom = vol_state[li].sum() + rev_vol_state[li][0]
            denom = jnp.where(denom < 1e-12, 1e-12, denom)
            right_bc_force_density = ramp_traction * (self.width * thickness[li]) / denom

            #right_bc_force_density = ramp_traction * (width * thickness[ri]) / (vol_state[ri].sum() + rev_vol_state[ri][0])
            
            #right_bc_mask = jax.device_get(self.right_bc_mask)
            #right_bc_indices = np.where(right_bc_mask)[0]
            
            mask = self.right_bc_mask
            right_bc_nodal_forces = right_bc_force_density * vol_state[li] * mask
            force = force.at[ri].add(-right_bc_nodal_forces.sum())


            #print("force: ", force)
        return force, inf_state


    #def solve_one_step(self, vals:Tuple[jax.Array, jax.Array, jax.Array, jax.Array, float]):
    def solve_one_step(self, vals):
        disp, vel, acc, inf_state, time, vol_state, rev_vol_state, thickness, time_step, width = vals
        print("ENTERING solve_one_step")

        # implementing a lower bound for thickness
        thickness = jnp.clip(thickness, 1e-3, None)  # Prevent zero/negative thickness
        #thickness = thickness  # Update the thickness in the problem object

        #print("t inside sol 1 st: ", thickness)
        #jax.debug.print("thickness inside sol 1 st: {}", thickness)
        #t_int = jax.device_get(thickness)
        #print("t_int in sol 1 step: ",t_int)
        
        #(disp, vel, acc, inf_state, time) = vals

        #jax.debug.print("thickness inside sol 1 st: {}", thickness)

        # TODO: Solve for stable time step
        #are we set on this tiem step?  Probs no need to increase this right no
        #time_step = 1.0e-9
        #print("self.left_bc_region: ", self.left_bc_region)

        
        #li = 0
        #moving to the setup_discretization function
        if self.prescribed_velocity is not None:
                bc_value = self.prescribed_velocity * time
                # Apply displacements bcs
                f = lambda x: 2.0 * bc_value / self.bar_length * x
                disp = disp.at[self.left_bc_region].set(f(self.pd_nodes[self.left_bc_region]))
                disp = disp.at[self.right_bc_region].set(f(self.pd_nodes[self.right_bc_region]))

        
        ###### think need to add something along the lines of the if statement above for self.prescribed_traction 
        ###### wasn't printing the below statements
        
        jax.debug.print("thickness in sol 1, before comp int f: {}", thickness)

        force, inf_state = self.compute_internal_force(disp, inf_state, time, vol_state, rev_vol_state, thickness, width)
        #print("inf_state in solve 1 step: ", inf_state)
        #jax.debug.print('thickness in 1 step after comp int for: {}', thickness)


        acc_new = force / self.rho
        vel = vel.at[:].add(0.5 * (acc + acc_new) * time_step)
        disp = disp.at[:].add(vel * time_step + (0.5 * acc_new * time_step * time_step))
        acc = acc.at[:].set(acc_new)

        #print("disp.shape: ",disp.shape)

        return (disp, vel, acc, inf_state, time + time_step, vol_state, rev_vol_state, thickness, time_step, width)


    def solve(self, 
              prescribed_velocity:Union[float, None], 
              prescribed_traction:Union[float, None], 
              vol_state,
              rev_vol_state,
              thickness,
              width,
              inf_state,
              max_time:float=1.0):

        '''
            changed how prescribed velocity and prescribed traction are set
            Solves in time using Verlet-Velocity
        '''
        #defining time before variable is referenced
        time = 0.0
        time_step = 1E-9

        # implementing a lower bound for thickness
        thickness = jnp.clip(thickness, 1e-3, None)  # Prevent zero/negative thickness
        #problem1.thickness = thickness  # Update the thickness in the problem object

        if prescribed_velocity is not None and prescribed_traction is not None:
            raise ValueError("Only one of prescribed_velocity or prescribed_traction should be set, not both.")
        
        if prescribed_velocity is None and prescribed_traction is None:
            raise ValueError("Either prescribed_velocity or prescribed_traction must be set.")

        jax.debug.print("thickness begin solve: {}",thickness)

        #self.prescribed_velocity = prescribed_velocity
        #self.prescribed_traction = prescribed_traction

        #initializing the fields locally
        disp = jnp.zeros_like(self.pd_nodes)
        vel = jnp.zeros_like(self.pd_nodes)
        acc = jnp.zeros_like(self.pd_nodes)
        #inf_state = self.influence_state
        #print("inf_state in solve: ",inf_state)


        force = jnp.zeros_like(self.pd_nodes)

        #:The node indices of the boundary region at the left end of the bar
        li = 0


        if prescribed_traction is not None:
            ramp_traction = self.smooth_ramp(time, t0=1.e-5, c=prescribed_traction)  

            denom = vol_state[li].sum() + rev_vol_state[li][0]
            denom = jnp.where(denom < 1e-12, 1e-12, denom)
            left_bc_force_density = ramp_traction * (self.width * thickness[li]) / denom

            #left_bc_force_density = ramp_traction * (self.width * thickness[li]) / (vol_state[li].sum() + rev_vol_state[li][0])

            #print("self.left_bc_mask: ", self.left_bc_mask)

            mask = self.left_bc_mask
            left_bc_nodal_forces = left_bc_force_density * vol_state[li] * mask
            force = force.at[li].add(-left_bc_nodal_forces.sum())

        #:The node indices of the boundary region at the right end of the bar
        ri = self.num_nodes - 1


        vals = (disp, vel, acc, inf_state, time, vol_state, rev_vol_state, thickness, time_step, width)
        #can't use while loop with dynamic bounds, max_time is set, bute vals[4] (time) changes in loop
        #vals = jax.lax.while_loop(lambda vals: vals[4] < max_time, self.solve_one_step, vals)

        
        #added the following lines to calculate the vals within solve_one_step in place of while loop above
        num_steps = int(max_time / time_step)

        def scan_body(carry, _):
            return self.solve_one_step(carry), None
        
        #jax.debug.print("inf_state in solve bef {}", inf_state)

        init_vals = (disp, vel, acc, inf_state, time, vol_state, rev_vol_state, thickness, time_step, width)
        final_vals, _ = jax.lax.scan(scan_body, init_vals, None, length=num_steps)

        jax.debug.print("thickness in solve aft {}", thickness)

    

        return final_vals

    def get_solution(self,disp):
        ''' Convenience function for retrieving displacement'''
        return disp

    def get_nodes(self):
        ''' Convenience function for retrieving peridynamic node locations'''
        return self.pd_nodes


def loss(thickness,problem1):
    '''
        Loss function for the optimization problem, which is to minimize the damage and thickness
        of the bar.
    '''
    problem1.allow_damage()

    # implementing a lower bound for thickness
    thickness = jnp.clip(thickness, 1e-4, None)  # Prevent zero/negative thickness
    #thickness = jnp.where(jnp.isnan(thickness), 1e-4, thickness)

    problem1.thickness = thickness  # Update the thickness in the problem object

    #print("problem1.thick begin loss: ", problem1.thickness)
    jax.debug.print("p1.thick begin loss: {}",problem1.thickness)
    jax.debug.print("thick begin loss: {}",thickness)


    #disp, vel, acc, inf_state = problem.reset_state(thickness)
    #disp = jnp.zeros(problem1.num_nodes)
    #vel = jnp.zeros(problem1.num_nodes)
    #acc = jnp.zeros(problem1.num_nodes)
    inf_state = problem1.influence_state
    width = 1

    # Update the model with the candidate thickness
    vol_state, rev_vol_state, inf_state = problem1.intermed_setup_disc(width, thickness)

    #problem1.setup_discretization(thickness, width)
    #vol_state, rev_vol_state = problem1.compute_partial_volumes(width, thickness)


    if problem1.prescribed_velocity is not None:
        prescribed_velocity = problem1.prescribed_velocity
        print("problem1.prescribed_velocity: ", problem1.prescribed_velocity)

    else:
        prescribed_velocity = None
    
    if problem1.prescribed_traction is not None:
        prescribed_traction = problem1.prescribed_traction
        print("problem1.prescribed_traction: ", problem1.prescribed_traction)

    else:
        prescribed_traction = None

    # Update the model with the candidate thickness
    #problem1.setup_discretization(thickness)

    #now run the problem with the locally defined fields and solve
    #problem1.reset(displacement, velocity, acceleration, influence_state, thickness)
    #problem1.setup_discretization(thickness) 

    disp, vel, acc, inf_state, time, vol_state, rev_vol_state, thickness, time_step, width = problem1.solve(prescribed_velocity=problem1.prescribed_velocity,prescribed_traction=problem1.prescribed_traction,vol_state=vol_state, rev_vol_state=rev_vol_state, thickness=thickness, width=width, inf_state=inf_state, max_time=1E-03)
    #ax.plot(problem1.get_nodes(), problem1.get_solution(disp), 'k.')
    #plt.show()

    jax.debug.print("thickness after solve in loss : {}", thickness)

    # feel like I need to update influence state here withn the problem here but not sure how to?
    #problem1.influence_state = inf_state

    #jax.debug.print("inf state loss fn: {}", inf_state)
    #jax.debug.print("vol state loss fn: {}", vol_state)

    damage = problem1.compute_damage(inf_state)
    #jax.debug.print("damage: {}", damage)



    mean_damage = damage.sum() / problem1.num_nodes
    max_damage =  damage.max()
    #jax.debug.print("mean_damage: {}", mean_damage)
    mean_thickness = thickness.sum() / problem1.num_nodes
    max_thickness = thickness.max()

    #says if loss is nan set to massive value
    #max_thickness = jnp.where(jnp.isnan(max_thickness), 1e10, max_thickness)
 
    thick_loss = mean_thickness / max_thickness

    #jax.debug.print("problem1.num_nodes: {}", problem1.num_nodes)
    #jax.debug.print("max_thickness: {}", max_thickness)

    #jax.debug.print("thick_loss: {}", thick_loss)
    #jax.debug.print("mean_damage: {}", mean_damage)
    #jax.debug.print("max_damage: {}", max_damage)

    loss_penalty = jnp.where(jnp.any(thickness < 0.01), 1E10, 0.0)
    
    loss = 0.5 * mean_damage + 0.5 * mean_thickness / max_thickness + loss_penalty
    #loss = 0.5 * max_damage + 0.5 * mean_thickness / max_thickness
    #loss = max_damage 

    # had tried to implement a bound to the loss function to solve nan issue
    '''
    def loss(thickness, problem1):
        if jnp.any(thickness < 0.01):
            return 1e10  # Large penalty for violating lower bound
    

    jax.debug.print("loss: {}", loss)
    '''

    # implementing a lower bound for thickness
    #thickness = jnp.clip(thickness, 1e-3, None)  # Prevent zero/negative thickness
    #problem1.thickness = thickness  # Update the thickness in the problem object
    #print("Updated thickness in problem1: ", problem1.thickness)


    #when add here still get nan values of thickness :/
    #thickness = jnp.where(jnp.isnan(thickness), 1e-4, thickness)
    #problem1.thickness = thickness  # Update the thickness in the problem object
    


    return loss


### Main Program ####
if __name__ == "__main__":

    #Define problem size
    fixed_length = 10.0 
    delta_x = 0.25
    fixed_horizon = 2.6 * delta_x

    #define load
    #pres_velo = 1.0
    pres_velo = None
    pres_traction = 5.0E8
    #pres_traction = None
    #max_time = 1E-02

    #Instantiate a 1d peridynamic problem with equally spaced nodes
    problem1 = PDJAX(bar_length=fixed_length,
                     density=7850.0,
                     bulk_modulus=200e9,
                     number_of_elements=int(fixed_length/delta_x), 
                     horizon=fixed_horizon,
                     thickness=1.0,
                     critical_stretch=1.0e-4,
                     prescribed_traction=pres_traction)
    
        #problem1.solve(max_time=1E3, prescribed_velocity=1.0)
    #


    ####### to forward run problem and plot displacement ################
    #######
    #disp, vel, acc, inf_state, _ = problem1.solve(prescribed_traction=pres_traction, prescribed_velocity=pres_velo,vol_state,max_time=1.0e-03)
    #fig, ax = plt.subplots()
    #ax.plot(problem1.get_nodes(), problem1.get_solution(disp), 'k.')
    #plt.show()

    ######
    ############################################################
    
    #problem1.introduce_flaw(0.0)
    #problem1.solve(max_time=1.0e-3, prescribed_velocity=1.0)
    #problem1.solve(max_time=1.0e-3, prescribed_velocity=1.0)
    #problem1.solve(max_time=1.0e-3, prescribed_traction=1E7)



    #
    #fig, ax = plt.subplots()
    #ax.plot(problem1.get_nodes(), problem1.get_solution(disp), 'k.')
    #plt.show()

    #'''
    ######## to run optimization loop and plot results ###############################################
    #############
    guess = jnp.ones(problem1.number_of_elements)
    print("guess thickness : ",guess)
    #thickness = 1
    result = jax.scipy.optimize.minimize(loss, guess, args=(problem1,), method='BFGS')
    optimized_thickness = result.x  # Optimized thickness from the result
    print("optimized_thickness: ",optimized_thickness)
    #problem1.setup_discretization(optimized_thickness)


    

    # Update the problem with the optimized thickness

    #printing the result of the optimization
    # Solve the problem using the optimized thickness
    disp, vel, acc, inf_state, time, vol_state, rev_vol_state, thickness, time_step, width = problem1.solve(
        prescribed_velocity=pres_velo, 
        prescribed_traction=pres_traction, 
        vol_state = vol_state,


        rev_vol_state = rev_vol_state,
        thickness = optimized_thickness,
        width = width,
        max_time=1.0E-03)
    
    fig, ax = plt.subplots()
    ax.plot(problem1.get_nodes(), problem1.get_solution(disp), 'k.')
    plt.show()
    

    # Plot the optimized thickness profile
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    # Plot the optimized thickness along the bar
    ax[0].plot(problem1.get_nodes(), optimized_thickness, 'b.-')
    ax[0].set_xlabel('Position along bar')
    ax[0].set_ylabel('Optimized Thickness')
    ax[0].set_title('Optimized Thickness Profile')

    # Plot the displacement field after optimization
    ax[1].plot(problem1.get_nodes(), problem1.get_solution(disp), 'k.-')
    ax[1].set_xlabel('Position along bar')
    ax[1].set_ylabel('Displacement')
    ax[1].set_title('Displacement After Optimization')

    plt.tight_layout()
    plt.show()
    #####################################
    #'''
    

