'''
Given an entangled object with finite componenets in 3-space the following program
calculates its JONES POLYNOMIAL
'''
import math
import numpy as np
import sys
import random
from copy import copy
import time
#from collections import Counter
from functions import *
from sympy.combinatorics import Permutation, PermutationGroup

closed=1 # if open, then assign value 0
''' 
================================================================
JONES POLYNOMIAL CALCULATION

In expected_Jones(Ch,n),
Ch (short for chain) signifies a 
list (of different components) of lists (of co-ordinates in any particular component) 
n signifies the no. of projections being considered (for averaging out our result)

In get_jones_poly(coords,proj_vector,inds),
coords signifies the co-ordinates of points on the curve in 3space
proj_vector signifies a particular direction of projection
inds signifies the connections among the points in the curve in concern.
'''

def expected_jones(Ch, n):
    """Calculate the expected Jones polynomial."""
    #print("#1")
    # Initialize indices among coordinates in 3-space
    inds = np.array([])
    ini = 0
    p = np.concatenate(Ch)
    
    # Generate indices for each chain in Ch
    for ch in Ch:
        ch = np.array(ch)
        sub_inds = np.concatenate([np.arange(1+ini, ch.shape[0]+ini), np.array([ini])])
        inds = np.concatenate([inds,sub_inds])
        ini = len(inds)	 
    inds = np.array([int(i) for i in inds])

    # Generate n vectors on the unit sphere uniformly
    points = []
    if n > 1:
        points = fibonacci_sphere(n)
        for i in range(n):
            points.append(get_random_proj())
    else:  # If n is not greater than 1, choose a specific vector
        points = [np.array([0.0,0.0,1.0])]
         #points=[get_random_proj()]

    JPOLY = []
    Jnone = 0

    # Calculate Jones polynomial for each projection vector
    for proj_vector in points:
        JPoly = []
        rj = get_jones_poly(p, proj_vector, inds)  # Jones polynomial along a specific projection vector
        if rj is not None:
            for base in rj:
                lenn = int((len(base)-1)/2)
                powers = range(-lenn, lenn+1, 1)
                poly = {}
                for i in range(len(base)):
                    if base[i] != 0:
                        poly[powers[i]] = base[i]
                JPoly.append(poly)
            JPoly.append(dfactor(len(JPoly)-1))
            Zpoly = {0:1}
            for i in range(len(JPoly)):
                Zpoly = J_mult(JPoly[i], Zpoly)  # Multiply Jones polynomials
            JPOLY.append(Zpoly)
        else:
            Jnone += 1

    # Addition of Jones polynomials
    Zpoly = {0:0}
    for i in range(len(JPOLY)):
        Zpoly = J_add(JPOLY[i], Zpoly)

    # Normalize the result by the number of successful Jones polynomial calculations
    n = n - Jnone
    Zpoly = J_smult(1./n, Zpoly)

    # Return the final averaged Jones polynomial
    return Zpoly

def get_jones_poly(coords, proj_vector, inds):
    """Calculate the Jones polynomial for a given set of coordinates and a projection vector."""
    #print("#2")
    #print(inds)
    global L1_Cr, L1_desired_crossings
    # Generate two vectors from the projection vector
    x, y = get_two_vec(proj_vector)
    
    # Project the coordinates onto the plane spanned by the two vectors
    proj1 = np.matmul(coords, np.stack([x, y], 1))
    proj = np.array([[round(elt[0], 3), round(elt[1], 3)] for elt in proj1])
    #print(proj)

    # Check for overlapping projections and return None if overlaps are detected
    Check_proj = np.unique([str(i) for i in proj], return_counts=True)
    if max(Check_proj[1]) > 2:
        return None

    # Calculate depth of each coordinate along the projection vector
    depth = np.matmul(coords, np.expand_dims(proj_vector, 1))[:, 0]
    
    # Generate boolean masks for different crossing properties
    bool_mask, over_or_under, u, right_or_left, proj, depth, inds = get_bool_overlap_etc(proj, inds, depth, start=0)
    
    range_var = bool_mask.shape[0]
    Cr = []
    for i in range(range_var):
        for j in range(i, range_var):
            if bool_mask[i, j]:
                Cr.append([i, j])
    
    desired_crossings = math.floor((sum(sum(bool_mask)) / 2) / 2)
    unused_crossings = [item for sublist in Cr[desired_crossings:] for item in sublist]
    #print(desired_crossings, unused_crossings)
    sub_matrices1_bm = process_crossings(bool_mask, inds, Cr, desired_crossings, unused_crossings)
    sub_matrices1_oou = process_crossings(over_or_under, inds, Cr, desired_crossings, unused_crossings)
    sub_matrices1_rol = process_crossings(right_or_left, inds, Cr, desired_crossings, unused_crossings)

    desired_crossings2 = len(Cr) - desired_crossings
    unused_crossings2 = [item for sublist in Cr[:desired_crossings] for item in sublist]
    #print(desired_crossings2, unused_crossings2)
    sub_matrices2_bm = process_crossings(bool_mask, inds, Cr, desired_crossings2, unused_crossings2)
    sub_matrices2_oou = process_crossings(over_or_under, inds, Cr, desired_crossings2, unused_crossings2)
    sub_matrices2_rol = process_crossings(right_or_left, inds, Cr, desired_crossings2, unused_crossings2)

    # Combine sub-matrices for the L1
    L1_bm = combine_sub_matrices(sub_matrices1_bm)
    #print("L1 BM:\n", L1_bm)
    L1_oou = combine_sub_matrices(sub_matrices1_oou)
    #print("L1 OOU:\n", L1_oou)
    L1_rol = combine_sub_matrices(sub_matrices1_rol)
    #print("L1 ROL:\n", L1_rol)

    # Combine sub-matrices for the L2
    L2_bm = combine_sub_matrices(sub_matrices2_bm)
    #print("L2 BM:\n", L2_bm)
    L2_oou = combine_sub_matrices(sub_matrices2_oou)
    #print("L2 OOU:\n", L2_oou)
    L2_rol = combine_sub_matrices(sub_matrices2_rol)
    #print("L2 ROL:\n", L2_rol)

    L1_inds = np.array([1, 0, 3, 4, 2])
    L2_inds = np.array([1, 2, 3, 0, 5, 6, 4])
    
    L1_PP = get_partial_poly(L1_bm, L1_oou, L1_rol, L1_inds)
    L2_PP = get_partial_poly(L2_bm, L2_oou, L2_rol, L2_inds)

    #print(L1_PP, L2_PP) 

    # Apply Reidemeister moves to reduce crossings
    #bool_mask = RM1(bool_mask, over_or_under, inds)
    #bool_mask = RM2(bool_mask, over_or_under, inds)
    #bool_mask = RM3(bool_mask, over_or_under, inds)

    # If the count of True values in the boolean mask is over 40, return None
    if np.count_nonzero(bool_mask) > 40: 
        return None #maybe split here
    else:
        #print('crossings',np.count_nonzero(bool_mask))
        K_list = []
        # Compute partial polynomial and writhe
        start_time=time.time()
        a = get_partial_poly(bool_mask, over_or_under, right_or_left, inds)
        end_time=time.time()
        #print('unbroken time', end_time-start_time)
        #print('Summed up', a)
        b = get_writhe(bool_mask, over_or_under, right_or_left)
        #print('Writhe', b)
        #b=0
        # Apply writhe to partial polynomial
        if b > 0:
            without_quarter = np.concatenate([np.zeros(6 * b), a * (-1) ** b])
            K = without_quarter
        elif b < 0:
            without_quarter = np.concatenate([a * (-1) ** (-b), np.zeros(-6 * b)])
            K = without_quarter
        else:
            K = a
        
        K_list.append(K)
        
        # Return the list of Jones polynomials
        return K_list

''' FIX POLY '''
def get_bool_overlap_etc(proj, inds, depth, start=0):
    #print("#3")
    # Create an empty matrix to represent whether each pair of edges intersect.
    bool_mask = np.zeros((len(inds),len(inds)), dtype=bool)

    # Create an empty matrix to represent, for each pair of intersecting edges, which edge is over the other.
    over_or_under = np.zeros((len(inds),len(inds)), dtype=bool)

    # Create an empty matrix to represent the intersection points of each pair of intersecting edges.
    u = np.zeros((len(inds),len(inds)), dtype=object)

    # Calculate differences between each pair of vertices.
    dif = proj[inds] - proj 
    #print(proj[inds])

    # Prepare a matrix used later to find intersections.
    flip_neg = np.stack([dif[:,1], -dif[:,0]], 1)



    # Iterate through each pair of edges.
    for i in range(len(inds)):
        for j in range(inds[i],len(inds),1):

            # Compute terms for the linear system to find intersections.
            pi=proj[i][0]*proj[inds[i]][1]-proj[i][1]*proj[inds[i]][0]
            pj=proj[j][0]*proj[inds[j]][1]-proj[j][1]*proj[inds[j]][0]
            Aij=np.array([[flip_neg[i][0],flip_neg[i][1],pi],[flip_neg[j][0],flip_neg[j][1],pj]])
            
            # Solve the linear system to find intersections using Cramer's rule.
            result=Cramer(Aij)

            # If an intersection is found
            if result != None:
                result=(round(result[0],3),round(result[1],3))

                # Determine the parameter of intersection points on the edges.
                xdif_i=proj[inds[i]][0]-proj[i][0]
                xdif_j=proj[inds[j]][0]-proj[j][0]
                if xdif_i!=0:
                    t=(result[0]-proj[i][0])/(proj[inds[i]][0]-proj[i][0])
                else:
                    t=(result[1]-proj[i][1])/(proj[inds[i]][1]-proj[i][1])
                if xdif_j!=0:
                    s=(result[0]-proj[j][0])/(proj[inds[j]][0]-proj[j][0])
                else:
                    s=(result[1]-proj[j][1])/(proj[inds[j]][1]-proj[j][1])

                # Buffer required for numerical process
                zero_val=1.e-2 

                # Check if intersection point lies within the edge segments.
                if zero_val<t<1-zero_val and zero_val<s<1-zero_val: 

                    # Record that the edges i and j intersect.
                    bool_mask[i,j]=True
                    bool_mask[j,i]=True

                    # Determine which edge is over the other at the intersection point.
                    zi=depth[i]+t*(depth[inds[i]]-depth[i])
                    zj=depth[j]+s*(depth[inds[j]]-depth[j])
                    if zi>zj:
                        over_or_under[i,j]=True # edge i over edge j
                    else:
                        over_or_under[j,i]=True # edge j over edge i

                    # Record the intersection point.
                    u[i,j]=result
                    u[j,i]=result   

    # For each pair of edges, determine which is to the right of the other.
    right_or_left = np.sum(np.expand_dims(flip_neg, 0)*np.expand_dims(dif, 1), -1)>0

    # If the graph is open, identify and ignore "false edges" that would cause an unintended cycle.
    if closed==0: 
        partition=how_many_loops(inds)[1]
        #print('components when open', partition)
        edges_2_ignore=[tup[1]-1 for tup in partition] 
        for i in edges_2_ignore:
            bool_mask[i,:]=False
            bool_mask[:,i]=False
    else:
        partition=how_many_loops(inds)[1]
        #print('components when closed', partition)
        edges_2_ignore=[]

    # If start is 0, introduce new vertices wherever there is one edge crossing more than one edge.
    if start ==0:

        dim=bool_mask.shape[0]

        new_vectors = []
        new_depth = []
        #Check if there are any edge with >=2 intersections
        for row in range(dim):
            new_vectors.append(list(proj[row]))
            new_depth.append(depth[row])
            if row not in edges_2_ignore:
                pts_on_segment=[]
                for col in range(dim):
                    if col not in edges_2_ignore:
                        if bool_mask[row,col]==True:
                            pts_on_segment.append(u[row,col])
                            
				# If the current edge has more than one intersection
                if len(pts_on_segment) > 1:
    				# Get the x and y differences between the starting and ending points of the edge
                    xdif_0 = proj[inds[row]][0] - proj[row][0]
                    xdif_1 = proj[inds[row]][1] - proj[row][1]
                    
                    # Create a dictionary to store the intersection points along the edge,
    				# keyed by the absolute magnitude of displacement from the starting point of the edge.
                    mag_vs_pt = {}  # magnitude of displacement vs point
                    for elt in pts_on_segment:
                        mag_vs_pt[abs(elt[0] - proj[row][0])] = elt
                    # Sort the intersection points along the edge by their displacement magnitudes.
                    mag_keys = list(mag_vs_pt.keys())
                    mag_keys.sort()
                    Ordered_points = []
                    for mag in mag_keys:
                        Ordered_points.append(mag_vs_pt[mag])
                        
   					# For every pair of consecutive intersection points along the edge,
    				# create a new point at their midpoint and add it to the list of vertices.
                    for i in range(len(Ordered_points) - 1):
                        p1 = Ordered_points[i]
                        p2 = Ordered_points[i + 1]
                        new_p = [(p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.]
                        
						# Calculate the depth of the new point as a weighted average of the depths of the two original points,
        				# based on its relative location between them.
                        if xdif_0 != 0:
                            new_dep = depth[row] + (new_p[0] - proj[row][0]) / xdif_0 * (depth[inds[row]] - depth[row])
                        else:
                            new_dep = depth[row] + (new_p[1] - proj[row][1]) / xdif_1 * (depth[inds[row]] - depth[row])
                            
        				# Add the new point and its depth to their respective lists.
                        new_vectors.append(new_p)
                        new_depth.append(new_dep)
                        
		# Initialize a list to store the indices of the first points of each partition in new_vectors
        NNV = []
        for chunk in partition:
    		# For each partition, get the first point
            first_pt = list(proj[chunk[0]])
    		# Get the index of the first point of the partition in new_vectors
            pt_ind = new_vectors.index(first_pt)
    		# Append the index to NNV
            NNV.append(pt_ind)

		# Append the length of new_vectors to NNV. This is likely done to include the last point in the iterations below.
        NNV.append(len(new_vectors))
        
		# Remove any duplicate indices from NNV and store the unique indices in NV
        NV = []
        for x in NNV:
            if x not in NV:
                NV.append(x)
                
		# Initialize inds as an empty numpy array. inds will store the indices of the points in the final ordering.
        inds = np.array([])

		# For each pair of consecutive elements in NV (representing a partition)
        for i in range(len(NV) - 1):
            # Create a numpy array with the indices of the points in the partition, and append the index of the first point to the end.
    		# This is likely done to form a closed loop.
            sub_inds = np.concatenate([np.arange(NV[i] + 1, NV[i + 1]), np.array([NV[i]])])
            # Concatenate the indices of the points in the partition to inds
            inds = np.concatenate([inds, sub_inds])
		# Convert the elements in inds to integers
        inds = np.array([int(i) for i in inds])
        
		# Convert new_vectors and new_depth to numpy arrays, and assign them to proj and depth, respectively
        proj = np.array(new_vectors)
        depth = np.array(new_depth)
        
		# Call how_many_loops function to determine the number of loops in inds, and store the partition indices in partitionp
        partitionp = how_many_loops(inds)[1]
        
		# Initialize newpoints as an empty list. newpoints will store the final points in their proper ordering.
        newpoints = []
        for tup in partitionp:
            # For each partition, create a list of points (with their corresponding depth), and append it to newpoints
            T = []
            for i in range(tup[0], tup[1]):
                T.append([proj[i][0], proj[i][1], depth[i]])
                newpoints.append(T)
                
			# Recursively call get_bool_overlap_etc function with the updated proj, inds, and depth
            bool_mask, over_or_under, u, right_or_left, proj, depth, inds = get_bool_overlap_etc(proj, inds, depth, start=1)
            #proj, inds, depth = add_point_to_edge(proj, inds, depth, bool_mask)
            
	# Return the updated boolean mask, overlap, u, orientation, projection, depth, and indices
    #print(inds)
    #print("bool","\n",bool_mask,"\n", "over or under","\n",over_or_under,"\n", "right or left","\n",right_or_left,"\n", "inds","\n",inds)
    return bool_mask, over_or_under, u, right_or_left, proj, depth, inds


'''BRACKET POLYNOMIAL'''
'''
Gets the characteristic polynomial
proj: should be a nx2 matrix of vertices
inds: describes the graph through the indices (should be a closed knot) (n size array)
depth: the depth of each vertex in three space (important for crossings)
edge2ignore: this value should indicate what edge is dropped (preventing a cycle)
'''

counter = 0
endpoints = None
loops = []
S = []
T = []

def classify_lists(data, endpoints):
    true_list = [sublist for sublist in data if not any(item in endpoints for item in sublist)]
    false_list = [sublist for sublist in data if any(item in endpoints for item in sublist)]
    return true_list, false_list

def get_partial_poly(bool_mask, over_or_under, right_or_left, inds):
    '''
    Check if there are any intersections
    '''

    # print(f"DEBUG: get_partial_poly called")
    # print(f"  bool_mask shape: {bool_mask.shape}")
    # print(f"  inds shape: {inds.shape}, values: {inds}")
    # print(f"  inds min: {min(inds)}, max: {max(inds)}")


    #print("#4")
    global counter, endpoints, loops
    if endpoints is None:
        endpoints = (min(inds), max(inds))

    if np.any(bool_mask):
        #print('IN YES')
        edge1 = np.argmax(np.any(bool_mask, 0))
        edge2 = np.argmax(bool_mask[edge1,:])
        #print("edge1: ", edge1, " edge2: ", edge2)

        # CRITICAL FIX: Check if edges are within bounds of inds array
        if edge1 >= len(inds) or edge2 >= len(inds):
            # print(f"DEBUG: Edge indices out of bounds - edge1={edge1}, edge2={edge2}, len(inds)={len(inds)}")
            # print(f"DEBUG: bool_mask shape={bool_mask.shape}, inds={inds}")
            
            # Remove the invalid crossing and continue
            bool_mask_fixed = bool_mask.copy()
            bool_mask_fixed[edge1, edge2] = False
            bool_mask_fixed[edge2, edge1] = False
            return get_partial_poly(bool_mask_fixed, over_or_under, right_or_left, inds)

        bool_mask1 = np.copy(bool_mask)
        bool_mask1[edge1, edge2] = False
        bool_mask1[edge2, edge1] = False

        '''
        We are going to create two new paths: path inds1 and path inds2
        '''
        inds1 = np.copy(inds)
        inds2 = np.copy(inds)
        reversed_ind = reversed_inds(inds)
        #print("reversed_ind", reversed_ind)

        #inds1 is easy because we just swap the destinations of edge1 and edge2
        inds1[edge1] = inds[edge2]
        inds1[edge2] = inds[edge1]

        #inds2 is substantially trickier
        inds2[edge1] = edge2
        first2flip = edge2
        
        # We have to reverse a few of the directions (hence a while loop)
        replacement = np.arange(inds.shape[0])
        #print('replacement',replacement)
        replacement[edge1] = edge2
        
        while first2flip != inds[edge1] and first2flip != inds[edge2]:
            inds2[first2flip] = reversed_ind[first2flip]
            replacement[first2flip] = reversed_ind[first2flip]
            first2flip = reversed_ind[first2flip]
        
        if first2flip == inds[edge1]:
            inds2[inds[edge1]] = inds[edge2]
            replacement[inds[edge1]] = edge1
        else:
            inds2[inds[edge2]] = inds[edge1]
            replacement[inds[edge2]] = edge1
        
        bool_mask2 = bool_mask1[replacement[:, None], replacement[None, :]]
        #print('bool 2',bool_mask2)
        right_or_left2 = right_or_left[replacement[:, None], replacement[None, :]]
        need_to_swap = replacement != np.arange(inds.shape[0])
        swap_mask = need_to_swap[:, None] != need_to_swap[None, :]
        right_or_left2[swap_mask] = np.logical_not(right_or_left2[swap_mask])
        over_or_under2 = over_or_under[replacement[:, None], replacement[None, :]]
        
        # We need the sub-polynomials / for every call: pp1 +1 pp2 -1
        counter += 1
        partial_poly1 = get_partial_poly(bool_mask1, np.copy(over_or_under), np.copy(right_or_left), inds1)

        counter -= 1
        partial_poly2 = get_partial_poly(bool_mask2, over_or_under2, right_or_left2, inds2)
        #print('PP1', partial_poly1, 'PP2', partial_poly2)
        
        # This part adds the polynomials
        # We first need to make sure all the polynomials are of the same degree
        # We do this by padding with zeros when necessary
        if partial_poly1.shape[0] > partial_poly2.shape[0]:
            width = int((partial_poly1.shape[0] - partial_poly2.shape[0]) / 2)
            partial_poly2 = np.concatenate([np.zeros([width]), partial_poly2, np.zeros([width])])
        
        if partial_poly2.shape[0] > partial_poly1.shape[0]:
            width = int((partial_poly2.shape[0] - partial_poly1.shape[0]) / 2)
            partial_poly1 = np.concatenate([np.zeros([width]), partial_poly1, np.zeros([width])])
        
        # We check for rightedness with aboveness
        if over_or_under[edge1, edge2] == right_or_left[edge1, edge2]:
            return (np.concatenate([partial_poly1, np.zeros([2])]) + np.concatenate([np.zeros([2]), partial_poly2]))
        else:
            return (np.concatenate([partial_poly2, np.zeros([2])]) + np.concatenate([np.zeros([2]), partial_poly1]))
    else:
        base = np.array([1])
        # We just count the number of loops. We subtract by 1 because one of the 
        # loops is broken up by the missing link
        num_loops = how_many_loops(inds)[0] - 1  # FOR CLOSED LOOPS
        for powi in range(num_loops):
            base = np.concatenate([-base, np.zeros([4])]) + np.concatenate([np.zeros([4]), -base])

        #print("Count", counter)
        #print("Endpoints", endpoints)

        loops = how_many_loops(inds)[1]
        #print("loops in a state", inds, loops)

        true_list, false_list = classify_lists(loops, endpoints)
        count_true = len(true_list)
        count_false = false_list
        a_factor = ([counter], [count_true], [count_false])
        #print("A Factor", a_factor)

        counter = 0

        return base

def get_writhe(bool_mask, over_or_under, right_or_left):
    #print("#5")

    # Create a triangular mask where only the upper-right elements are True.
    # This removes the diagonal and lower-left elements which represent duplicate intersections or self-intersections.
    bool_mask[np.arange(bool_mask.shape[0]).reshape((-1,1)) <= np.arange(bool_mask.shape[0]).reshape((1, -1))] = False

    # Calculate the writhe: 
    # 1. Determine where over_or_under and right_or_left match (i.e., where the path goes "over" and to the "right" or "under" and to the "left")
    # 2. Convert this boolean matrix to integer (True -> 1, False -> 0)
    # 3. Multiply by 2 and subtract 1 to get values of 1 where the paths match and -1 where they don't
    # 4. Sum these values over the elements where bool_mask is True, i.e., over the upper-right elements representing valid intersections
    return np.sum(2 * np.int32(over_or_under == right_or_left)[bool_mask] - 1)

def Loops(inds):
	count=0
	L=[]
	old=[]
	for k in inds :
		#print(k)
		if k not in old:
			x=1
			cyc=[k]
			old.append(k)
			while x>0:
				try:
					cyc.append(inds[k])
					k=inds[k]
					old.append(k)
					if k==cyc[0]:
						x=0
				except:
					x=0
			count+=1		
			L.append(cyc)
	#print(old)		
	return count, L	

sigma = [[0,0],[3,3]]
#sigma=[[4,0],[0,3],[3,2],[7,5]]
def compute_final(S, T, sigma):
    #print('S', S)
    #print('T', T)
    final = []
    for s in S:
        for t in T:
            segment_loops = {}
            key = tuple(s[2][0][0]) if isinstance(s[2][0][0], list) else s[2][0][0]
            segment_loops[key] = s[2][0][1]
            for k in sigma:
                segment_loops[k[0]] = k[1]
            elt = (sum(s[0] + t[0]), sum(s[1] + t[1]), Loops(segment_loops)[0])
            if elt not in final:
                final.append(elt)
    #print("FINAL FROM SPLIT",final)
    return final

# Function to extract arcs and points
def extract_arcs_and_points(crossings, desired_crossings):
    selected_crossings = crossings[:desired_crossings]
    crossing_points = []
    arcs = []
    for crossing in selected_crossings:
        crossing_points.append(crossing)
        for arc in crossing:
            point_pair = [arc, arc + 1]
            if point_pair not in arcs:
                arcs.append(point_pair)
    return crossing_points, arcs

# Function to form connections
def form_connections(arcs, crossings):
    flat_points = [p for sublist in arcs for p in sublist]
    sorted_points = sorted(list(set(flat_points)))
    for i in range(len(sorted_points) - 1):
        connection = [sorted_points[i], sorted_points[i + 1]]
        if abs(connection[1] - connection[0]) <= 1 and connection not in arcs and connection not in crossings:
            arcs.append(connection)
    return arcs

# Function to find continuous ranges
def find_continuous_ranges(inds):
    #print(inds)
    if not inds:
        return []
    start = inds[0]
    end = inds[0]
    ranges = []
    for i in range(1, len(inds)):
        if inds[i] == end + 1:
            end = inds[i]
        else:
            ranges.append((start, end))
            start = inds[i]
            end = inds[i]
    ranges.append((start, end))
    #print(ranges)
    return ranges

# Main processing function
def process_crossings(bool_mask, inds, crossings, desired_crossings, unused_crossings):
    inds = [i for i in inds if i not in unused_crossings]
    #print(unused_crossings,inds)
    crossing_points, arcs = extract_arcs_and_points(crossings, desired_crossings)
    arcs = form_connections(arcs, unused_crossings)
    continuous_ranges = sorted(find_continuous_ranges(inds))
    sub_matrices = []
    for r1 in continuous_ranges:
        for r2 in continuous_ranges:
            sub_matrix = bool_mask[r1[0]:r1[1] + 1, r2[0]:r2[1] + 1]
            sub_matrices.append(sub_matrix)
    return sub_matrices
    
# Function to combine sub-matrices
def combine_sub_matrices(sub_matrices):
    order = list(range(len(sub_matrices)))
    num_to_combine = int(math.sqrt(len(order)))

    combined_by_cols = []
    while sub_matrices:
        combined_col = np.hstack([sub_matrices.pop(0) for _ in range(num_to_combine)])
        combined_by_cols.append(combined_col)

    return np.vstack(combined_by_cols)