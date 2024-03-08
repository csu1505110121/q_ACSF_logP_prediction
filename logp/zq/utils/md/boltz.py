#!/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import mdtraj as mdtraj
from sklearn.cluster import KMeans

import math

from logp.zq.utils.md.load_prop import load_energy

K_BOLTZ = 1.38*10**-23
TEMP = 300
AVOGADRO = 6.02*10**23
CAL2J = 4.184

def get_xyz(filename,idx,output):
	pdbs = []
	with open(filename,'r') as f:
		while True:
			line = f.readline()
			if not line:
				break
			else:
				if 'MODEL' in line:
					pdb = []
					pdb.append(line)
					for i in range(9999):
						line = f.readline()
						if 'ENDMDL' not in line:
							pdb.append(line)
						else:
							pdb.append(line)
							break
					pdbs.append(pdb)
	dump_pdb(pdbs[idx],output)
	
def dump_pdb(pdb,filename):
	with open(filename,'w') as f:
		for line in pdb:
			f.write(line)

def boltz(prefix,index_list,energy_key='v'):
	"""
	$$ p_i  = \frac{e^{-\epsilon_i /kT}}{\sum_{j=1}^{M=len(index_list)}e^{-\epsilon_j/kT}} $$


	# energy_key could be 'v' : potential energy (default one)
							'k': kinetic energy
							'e': total energy
	# Return the probability and dump corresponding structure to the file named index.pdb
	"""	
	prob_list = []
	xyz_dump = prefix+"/output.pdb"
	ene = load_energy(prefix,energy_key)
	ene_list = []
	#print('ENE OF INDEXED IS {}'.format(ene))
	for idx in index_list:
		output = prefix+'/'+str(idx)+'.pdb'  # need to set prefix
		get_xyz(xyz_dump,idx,output)
		ene_list.append(float(ene[idx]))
	
	#print('ENE LIST IS {}'.format(ene_list))
	ene_min = min(ene_list)
	delta_ene = [ene - ene_min for ene in ene_list]
	#print('DELTA ENE IS {}'.format(delta_ene))

	for ene in delta_ene:
		prob = math.exp(-float(ene)*1000/AVOGADRO/(K_BOLTZ*TEMP))
		prob_list.append(prob)

	#print('PROB IS {}'.format(prob_list))
	
	return np.array(prob_list)/sum(prob_list)

def get_class_center(prefix,smile,cluster_index):
	filename = prefix+str(smile)+'/output.pdb'
	traj = mdtraj.load(filename)

	traj_slice = traj[cluster_index]
	# define heavy atoms to be selected
	atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']
	
	distances = np.empty((traj_slice.n_frames, traj_slice.n_frames))
	
	for i in range(traj_slice.n_frames):
		distances[i] = mdtraj.rmsd(traj_slice, traj_slice, i, atom_indices = atom_indices)

	# define the similarity between any i and j structure
	# $$ s_{ij} = exp^{-\beta d_{ij} / d_{scale} }$$
	# more details could be found in https://mdtraj.org/1.9.4/examples/centroids.html?highlight=cluster
	beta = 1.0
	index = np.exp(-beta * distances/distances.std()).sum(axis=1).argmax()
	
	#print(index)
	return cluster_index[0][index]


def get_similarity(prefix,smile):
	filename = prefix + str(smile)+'/output.pdb'
	traj = mdtraj.load(filename)

	# define heavy atoms to be selected
	atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']

	distance = np.empty((traj.n_frames, traj.n_frames))

	for i in range(traj.n_frames):
		distance[i] = mdtraj.rmsd(traj, traj, i, atom_indices = atom_indices)

	return distance



def cluster_traj(prefix,smile,save_super=False,cmethods='KMeans',n_clusters=3):
	"""
	args :
		Why n_clusters set to be 3?
		1) considering the subsequently computational cost for generate SFs
		2) a proverb says "not to do anything more than three times"
	return:
		index of frames in each clusters
	"""
	filename = prefix+str(smile)+'/output.pdb'
	traj = mdtraj.load(filename)

	atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol !='H']
	
	traj_superposed = traj.superpose(traj,frame=0,atom_indices=atom_indices)
	if save_super:
		filename_superposed = prefix+str(smile)+'/output_superposed.pdb'
		traj_superposed.save_pdb(filename_superposed)
	
	n_frames = traj_superposed.n_frames
	xyz_list = traj_superposed.xyz.reshape(n_frames,-1)
	#print(xyz_list.shape)
	if cmethods == 'KMeans':
		kmeans = KMeans(n_clusters=n_clusters,random_state=9).fit(xyz_list)

	clusters = kmeans.labels_
	
	results = []
	
	for i in range(n_clusters):
		index = np.where(clusters==i)
		results.append(index)

	return results
	
	

if __name__ == '__main__':
	filename = '/home/zhuqiang/LogP_prediction/logp_prediction/logp/DATASETS/MD/star/OC1=CC=CC=C1/output.pdb'
	get_xyz(filename,1,str(1)+'.pdb')
