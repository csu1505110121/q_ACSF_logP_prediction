#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from logp.sf_behler.sf import sum_G1, sum_G2
from logp.zq.io.base import list_loader

def is_normal(filename):
	with open(filename,'r') as f:
		lines = f.readlines()
		if 'Normal termination' in lines[-1]:
			return True
		else:
			return False

def load_gau(filename,prop):
	"""
	input:
	    - the filename of GAUSSIAN 09 (tested)
		- the filename of GAUSSIAN 16 need to revise the atomic charges
	output:
	    - xyz (in the unit of angstrom)
	    - charge,q  (in the unit of e)
	"""
	if prop == 'xyz':
		xyzs = []
		if is_normal(filename):
			with open(filename,'r') as f:
				while True:
					line = f.readline()
					if not line:
						break
					else:
						if 'Standard orientation' in line:
							idx = []
							xyz = []
							# skip the lines
							# ---------------------------------------------------------------------
							# Center     Atomic      Atomic             Coordinates (Angstroms)
							# Number     Number       Type             X           Y           Z
							# ---------------------------------------------------------------------
							for i in range(4):
								line = f.readline()
							for i in range(999):
								line = f.readline()
								if '-------' in line:
									xyzs.append(xyz)
									break
								else:
									data = line.split()
									idx.append(int(data[1]))
									xyz.append([float(data[3]),float(data[4]),float(data[5])])
		else:
			print('Not Terminated Normally!')
		return np.array(idx), np.array(xyzs[-1])

	elif prop == 'charge':
		qs = []
		if is_normal(filename):
			with open(filename,'r') as f:
				while True:
					line = f.readline()
					if not line:
						break
					else:
						# only works for GAUSSIAN 09
						if 'Mulliken atomic charges:' in line:
							q = []
							# skip the line
							#             1
							line = f.readline()
							for i in range(999):
								line = f.readline()
								if 'Sum of Mulliken atomic charges' in line:
									qs.append(q)
									break
								else:
									data = line.split()
									q.append(float(data[2]))
		else:
			print('Not Terminated Normally!')
		#print(qs)
		# consistent with the api load_prop
		# charge in the type of list
		return qs[-1]



def generator_feat_gaussian(prefix,cas_no,prop,params,weights='charge',max_out=False):
	
	dataset = {'elems':[],'coord':[],'fps':[],'logp':[]}

	root_gaussian = prefix+cas_no+'.out'
	elem_idx, xyz = load_gau(root_gaussian,'xyz')
	q = load_gau(root_gaussian,'charge')

	if weights == 'charge':
		weights = q
	elif weights == None:
		weights = None

	rad = params['G1']
	ang = params['G2']

	n_feats_G1 = len(rad['eta'])
	n_feats_G2 = len(ang['lamb']) * len(ang['eta'])

	fps_G1 = np.zeros((len(elem_idx),n_feats_G1), dtype=np.float32)
	fps_G2 = np.zeros((len(elem_idx),n_feats_G2), dtype=np.float32)

	for i, eta in enumerate(rad['eta']):
		datas = sum_G1(xyz, eta=eta, Rs=rad['Rs'][i], weights=weights)
		for j, data in enumerate(datas):
			fps_G1[j][i]=data

	feature = 0
	for i, lamb in enumerate(ang['lamb']):
		for j, eta in enumerate(ang['eta']):
			datas = sum_G2(xyz, lamb=lamb, eta=eta, xi=ang['xi'][j],weights=weights)
			feature +=1
			for k,data in enumerate(datas):
				fps_G2[k][feature-1] = data

	elemsidx = np.array(elem_idx, np.int32)
	coord = np.array(xyz, np.float32)
	fps = np.concatenate((fps_G1, fps_G2), axis=-1)
	prop = float(prop)

	dataset['elems'] = elemsidx
	dataset['coord'] = coord
	dataset['fps']   = fps
	dataset['logp']  = prop

	return dataset


if __name__ == '__main__':
	q = load_gau('/home/zhuqiang/LogP_prediction/logp_prediction/logp/DATASETS/Rate_Constant/logs/99-96-7-D.out','charge')
