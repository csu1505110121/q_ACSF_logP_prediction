#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import distance_matrix
from logp.sf_behler.dump import dump_pdb


def get_charge(mol):
	AllChem.ComputeGasteigerCharges(mol)
	contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') \
					for i in range(mol.GetNumAtoms())]
	return contribs


def distMatrix(coords):
	return distance_matrix(coords,coords)


def f_c(R_ij,Rc=6.0):
	"""
	R_ij: distance between atom i and j;
	Rc: cutoff distance assigned in advance
	"""
	if R_ij <= Rc:
		return 0.5*(math.cos(R_ij*math.pi/Rc) + 1 )
	else:
		return 0

def G1(R_ij,eta,Rs):
	"""
	R_ij: distance between atom i and j;
	Rs: the average distance of the surrounding atoms;
	f_c: function of cutoff;
	eta: parameter of the gaussian function
	"""
	fc = f_c(R_ij)
	return math.exp(-eta*(R_ij-Rs)**2)*fc

def sum_G1(coords,eta=2,Rs=0,weights=None):
	# calculate the distance matrix between any two points
	distMat = distMatrix(coords)
	# variable for molecule symmetry function
	# number of atoms of the molecule
	# eg: for water, 3 for O, H1, H2, respectively
	molecule_sum = []
	for i in range(coords.shape[0]):
		# variable for atom symmetry function
		atom_sum = []

		if weights ==None:
			for j in range(coords.shape[0]):
				if i != j:
					atom_sum.append(G1(distMat[i][j],eta,Rs))
			molecule_sum.append(sum(atom_sum))
		else:
			for j in range(coords.shape[0]):
				if i !=j:
					atom_sum.append(G1(distMat[i][j],eta,Rs) * weights[j])
			molecule_sum.append(sum(atom_sum))
				
	return molecule_sum
				 
	#for i in range()

# 计算cutoff中任意3个原子形成的角度
def angle(coord_i,coord_j,coord_k):
	"""
	return the angle formed between index j-i-k 
			center around atom i
			in the unit of Rad 
	"""
	v1 = coord_j - coord_i
	v2 = coord_k - coord_i
	len_v1 = np.linalg.norm(v1)
	len_v2 = np.linalg.norm(v2)
	# 为了避免程序出现-1.00000000002这样的现象
	acos_v = np.clip(np.dot(v1,v2)/(len_v1*len_v2), -1.0, 1.0)
	return math.acos(acos_v)


def G2(R_ij,R_ik,R_jk,theta_ijk,lamb,eta,xi):
	"""
	R_ij, R_ik, R_jk denotes the distance formed between atom i,j,k, respectively.
	theta_ijk: angle formed between atom j-i-k center at i
	eta: parameter of the gaussian function
	xi: a parameter
	"""
	fc_ij = f_c(R_ij)
	fc_ik = f_c(R_ik)
	fc_jk = f_c(R_jk)
	#print("fc function:", fc_ij,fc_ik,fc_jk)
	G2 = 2**(1 - xi) * (1 + lamb * math.cos(theta_ijk))**xi * \
		 math.exp(-eta * (R_ij**2 + R_ik**2 + R_jk**2)) * \
		 fc_ij * fc_ik * fc_jk
	return G2

def sum_G2(coords,lamb=1,eta=2,xi=1,weights=None):
	# calculate the distance matrix between any two points
	distMat = distMatrix(coords)
	molecule_sum = []
	for i in range(coords.shape[0]):
		atom_sum = []
		# variable for atom symmetry function
		if weights ==None:
			for j in range(coords.shape[0]):
				for k in range(coords.shape[0]):
					if i != j and i != k and j != k:
						theta_ijk = angle(coords[i],coords[j],coords[k])
						#print(i,j,k,theta_ijk)
						#print(distMat[i][j],distMat[i][k],distMat[j][k])
						atom_sum.append(G2(distMat[i][j],distMat[i][k],distMat[j][k],\
										theta_ijk,\
										lamb,eta,xi))
		else:
			for j in range(coords.shape[0]):
				for k in range(coords.shape[0]):
					if i != j and i !=k and j != k:
						theta_ijk = angle(coords[i], coords[j], coords[k])
						
						atom_sum.append(G2(distMat[i][j],distMat[i][k],distMat[j][k],\
										theta_ijk,\
										lamb,eta,xi) * (weights[j]*weights[k]))
		#			print(atom_sum)
		molecule_sum.append(sum(atom_sum))
	return molecule_sum
	


if __name__ == '__main__':
	from g3D import XYZ_dict

	smile = 'c1ccccc1'
	mol = Chem.MolFromSmiles(smile)
	xyz,elements = XYZ_dict(mol,method='UFF')
	#print(xyz.shape)
	#print(xyz[0][0],xyz[0][1],xyz[0][2])
	dump_pdb('benzene.pdb',xyz,elements)
	#distMat = distMatrix(xyz)
	#print(distMat)
	sumG1 = sum_G1(xyz)
	print(sumG1)
	sumG2_p = sum_G2(xyz,lamb=1)
	print(sumG2_p)
	sumG2_n = sum_G2(xyz,lamb=-1)
	print(sumG2_n)
