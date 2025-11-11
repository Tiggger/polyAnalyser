import math
import numpy as np
import sys
import random
from copy import copy
import time

store_part = {}

# Picks a random point from the surface of a unit sphere.
def get_random_proj():
	b = 0
	while np.sum(b*b)>1 or np.sum(b*b)<1e-4:
		b = 2*np.random.uniform(size=3)-1
	return b/np.sqrt(np.sum(b*b))


# Chooses n = samples points from the surface of a unit sphere.
def fibonacci_sphere(samples):
	points = []
	phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
	for i in range(samples):
		y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
		radius = math.sqrt(1 - y * y)  # radius at y
		theta = phi * i  # golden angle increment
		x = math.cos(theta) * radius
		z = math.sin(theta) * radius
		points.append((x, y, z))
	return points



# Given a normal vector, finds the corresponding 
# plane selecting 2 orthonormal basis vectors representative  of the plane
def get_two_vec(proj_vec): 
	a = np.zeros([3])
	proj_vec=np.array(proj_vec)
	while np.sum(a*a) < 1e-4:
		a = np.random.normal(size=[3])
		a = a - proj_vec*np.sum(a*proj_vec)/np.sum(proj_vec*proj_vec)
	a /= np.sqrt(np.sum(a*a))
	b = np.array([
				  a[1]*proj_vec[2]-a[2]*proj_vec[1], 
				  -a[0]*proj_vec[2]+a[2]*proj_vec[0],
				  a[0]*proj_vec[1]-a[1]*proj_vec[0]
				  ])	#cross product with a and proj
						#orthogonality of a and b can be checked using np.dot(a,b)
	return a, b



#Count how many loops described by the indices
def how_many_loops(inds):
	remaining = np.ones(inds.shape, dtype=np.bool)
	#print('rem',remaining)
	count=0
	L=[]
	while(np.any(remaining)>0):
		next_ind = np.argmax(remaining)
		cyc=[next_ind]
		#print('argmax',np.argmax(remaining))
		length=0
		while(remaining[next_ind]):
			remaining[next_ind] = False
			next_ind = inds[next_ind]
			if next_ind not in cyc:
				cyc.append(next_ind)
			length+=1
		count+=1
		#cyc=[next_ind,next_ind+length]
		L.append(cyc)
	return count, L   



# Reverse the direction of every edge
# i.e. if vertex a went to vertex b, now vertex b goes to vertex a
def reversed_inds(inds):
	return np.arange(inds.shape[0])[np.argsort(inds)]



#This generates a matrix indicating which edges are crossing, where they are crossing, 
#which line is on top, etc.
''' CRAMER'S RULE 2x2'''
def det2x2(A):
	assert A.shape == (2,2)
	return A[0][0]*A[1][1] - A[0][1]*A[1][0]

def Cramer(A):
	assert A.shape == (2,3)
	D = det2x2(A[:,:2])
	if D == 0:
		return
	Dx = det2x2(A[:,[2,1]])
	Dy = det2x2(A[:,[0,2]])
	return Dx*1.0/D, Dy*1.0/D



# Reidemeister MOVES

def RM1(bool_mask,  over_or_under, inds):
	#print(bool_mask)
	partition=how_many_loops(inds)[1]
	#print(partition)
	Bin=[]
	for tup in partition:
		for edge1 in tup:
			for edge2 in tup:#tup[tup.index(edge1)+1:]+tup[0:tup.index(edge1)]:
				#print(edge1,edge2)
				if bool_mask[edge1,edge2]==True and edge1 not in Bin:
					#print(edge1,edge2)
					Bin=Bin+[edge1,edge2]
					arc1=tup[tup.index(edge1)+1:tup.index(edge2)]
					arc2=tup[tup.index(edge2)+1:]+tup[0:tup.index(edge1)]
					#print(arc1,arc2)
					Eval1=[]
					Eval2=[]
					for i in arc1:
						#print(i,bool_mask[i,:])
						if np.any(bool_mask[i,:]):
							j=np.argmax(bool_mask[i,:])
							#print(j)
							if j not in arc1:
								#print(i,j,(over_or_under[i,j]))
								Eval1.append(np.int32(over_or_under[i,j]))
					for i in arc2:
						if np.any(bool_mask[i,:]):
							j=np.argmax(bool_mask[i,:])
							if j not in arc2:
								Eval2.append(np.int32(over_or_under[i,j]))
												
					#print('eval',Eval1,Eval2)
					if len(Eval1)==0 or len(Eval2)==0:
						bool_mask[edge1,edge2]=False
						bool_mask[edge2,edge1]=False
					else:
						flag=0
						#while flag==0:
						if len(Eval1)%2 == 1:
							if len(Eval2)%2==1:
								flag=1
							else:
								for a in range(0,len(Eval2),2):
								#print(i,i+1, Eval[i],Eval[i+1])
									if Eval2[a]!=Eval2[a+1]:
										flag=1
										break
									else:
										flag=0
						else:
							for a in range(0,len(Eval1),2):
								#print(i,i+1, Eval[i],Eval[i+1])
								if Eval1[a]!=Eval1[a+1]:
									if len(Eval2)%2==1:
										flag=1
										break
									else:
										for b in range(0,len(Eval2),2):
										#print(i,i+1, Eval[i],Eval[i+1])
											if Eval2[b]!=Eval2[b+1]:
												flag=1
												break
											else:
												flag=0
								else:
									flag=0
						if flag==0: 
							#print('0 flag')
							bool_mask[edge1,edge2]=False
							bool_mask[edge2,edge1]=False
						#flag=1
	#print('bool_mask',bool_mask)		
	#print('RM1 HI')	
	return bool_mask

	
def RM2(bool_mask, over_or_under,inds):
	partition=how_many_loops(inds)[1]
	Bin=[]
	#print(inds)
	#print('Bin',len(partition),partition)
	for tup1 in partition:
		for tup2 in partition: 
			for edge1 in tup1:
				for edge2 in tup2:
					if bool_mask[edge1,edge2]==True and edge1 not in Bin:
						#print('jjj',edge1,edge2)
						for edge3 in tup1[tup1.index(edge1)+1:]+tup1[0:tup1.index(edge1)]:
							for edge4 in tup2[tup2.index(edge2)+1:]+tup2[0:tup2.index(edge2)]:
								if bool_mask[edge3,edge4]==True and edge3!=edge2 and edge4 !=edge1:
									#print('yo',edge1,edge2,edge3,edge4)
									Bin=Bin+[edge1,edge2,edge3,edge4]
									if tup1.index(edge1)<tup1.index(edge3):
										arc1=tup1[tup1.index(edge1):tup1.index(edge3)+1]
									else:
										arc1=tup1[tup1.index(edge1):]+tup1[0:tup1.index(edge3)+1]
									if tup2.index(edge2)<tup2.index(edge4):
										arc2=tup2[tup2.index(edge2):tup2.index(edge4)+1]
									else:
										arc2=tup2[tup2.index(edge2):]+tup2[0:tup2.index(edge4)+1]
									#print(arc1,arc2)
									Eval1=[]
									warn=0
									for i in arc1:
										#print(i,bool_mask[i,:])
										if np.any(bool_mask[i,:]):
											j=np.argmax(bool_mask[i,:])
											if j in arc2:
												#print('SAME',i,j,(over_or_under[i,j]))
												Eval1.append(np.int32(over_or_under[i,j]))
											else:
												#print('WARNNNNN',i,j)
												warn=1
									#print('eval',Eval1)
									flag=0
									if len(Eval1)%2 == 1:
										flag=1
									elif len(Eval1)==0:
										flag=0
									else:
										for a in range(len(Eval1)-1):
										#print(i,i+1, Eval[i],Eval[i+1])
											try:
												test_1=Eval[a+1]
												if Eval1[a]==Eval1[a+1]:
													try:
														Eval1=Eval1[0:a]+Eval1[a+2:]
													except:
														try:
															Eval1=Eval1[0:a]
														except:
															Eval1=Eval1[a+2:]
											except:
												break
												
										if len(Eval1)%2 == 1:
											flag=1
										elif len(Eval1)==0:
											flag=0
										else:
											for a in range(0,len(Eval1),2):
												#print(i,i+1, Eval[i],Eval[i+1])
												if Eval1[a]!=Eval1[a+1]:
													flag=1
													break
												else:
													flag=0

									if flag==0 and warn!=1: 
										for a in arc1:
											for b in arc2:
												bool_mask[a,b]=False
												bool_mask[b,a]=False
	return bool_mask

# Reidemeister MOVE 3 (RM3)
def RM3(bool_mask, over_or_under, inds):
    partition = how_many_loops(inds)[1]
    Bin = []
    #print(bool_mask)

    for tup in partition:
        for i in range(len(tup)):
            for j in range(i+1, len(tup)):
                for k in range(j+1, len(tup)):
                    # Check if the crossings i-j, j-k, and i-k all exist
                    if bool_mask[tup[i], tup[j]] and bool_mask[tup[j], tup[k]] and bool_mask[tup[i], tup[k]]:
                        # Check if the crossing i-j is under, j-k is over, and i-k is under (one possible orientation for RM3)
                        if not over_or_under[tup[i], tup[j]] and over_or_under[tup[j], tup[k]] and not over_or_under[tup[i], tup[k]]:
                            #print("Before RM3: ", bool_mask)
                            # Perform RM3
                            bool_mask[tup[i], tup[j]], bool_mask[tup[j], tup[i]] = False, False
                            bool_mask[tup[j], tup[k]], bool_mask[tup[k], tup[j]] = False, False
                            bool_mask[tup[i], tup[k]], bool_mask[tup[k], tup[i]] = True, True
                            Bin = Bin + [tup[i], tup[j], tup[k]]
                            #print("After RM3: ", bool_mask)
                        # Check if the crossing i-j is over, j-k is under, and i-k is over (the other possible orientation for RM3)
                        elif over_or_under[tup[i], tup[j]] and not over_or_under[tup[j], tup[k]] and over_or_under[tup[i], tup[k]]:
                            #print("Before RM3: ", bool_mask)
                            # Perform RM3
                            bool_mask[tup[i], tup[j]], bool_mask[tup[j], tup[i]] = False, False
                            bool_mask[tup[j], tup[k]], bool_mask[tup[k], tup[j]] = False, False
                            bool_mask[tup[i], tup[k]], bool_mask[tup[k], tup[i]] = True, True
                            Bin = Bin + [tup[i], tup[j], tup[k]]
                            #print("After RM3: ", bool_mask)

    return bool_mask

def simplification(BM,over_or_under,inds):
	#print('yep')
	BM=RM3(BM, over_or_under,inds)
	BM=RM2(BM, over_or_under,inds)
	BM=RM1(BM, over_or_under,inds)
	return BM		

def max_len(Ch):
	l=0
	for ch in Ch:
		l=max([l,len(ch)])
	return l

def J_mult(P,Q):
	Z={}
	for i in P:
		for j in Q: 
			try:
				Z[i+j]+=P[i]*Q[j]
			except:
				Z[i+j]=P[i]*Q[j]
	return Z

def J_add(P,Q):
	Z=P
	for j in Q: 
		try:
			Z[j]+=Q[j]
		except:
			Z[j]=Q[j]
	return Z

def J_smult(a,P):
	Z={}
	for i in P:
		Z[i]=a*P[i]
	return Z
	
def dfactor(N):
	dpoly={0:1}
	for i in range(N):
		dpoly=J_mult(dpoly,{-2:-1,2:-1})
	return dpoly





