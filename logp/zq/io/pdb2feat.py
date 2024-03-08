#!/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from logp.sf_behler.sf import sum_G1, sum_G2
from logp.zq.io.base import list_loader
from logp.zq.utils.g3D import pdbinfo
from logp.zq.utils.md.load_prop import load_prop

# get pdb coordinate and element info
from logp.zq.utils.g3D.pdbinfo import pdbinfo
# load prop such charge, sigma, epsilon info from md section
from logp.zq.utils.md.load_prop import load_prop

from logp.zq.utils.md.boltz import cluster_traj, get_class_center, boltz

def load_feat_from_csv_md(datalist,feat_num,**kwargs):
	
	@list_loader(feat_num=feat_num)
	def _feat_gen(data):
		elems = data[0]
		coord = data[1]
		fps = data[2]
		logp = data[3]

		data = {'elems':elems, 'coord':coord, 'fps':fps, 'logp':logp}

		return data
	return _feat_gen(datalist,**kwargs)

def generator_feat_md(prefix,smile,logp,params,weights='charge',max_out=False):
	dataset = {'elems':[],'coord':[],'fps':[],'logp':[],\
				'max_elems':[],'max_coord':[],'max_fps':[],'max_logp':[]}
	#dataset_max = {'elems':[],'coord':[],'fps':[],'logp':[]}

	root_md = prefix+smile
	charge = load_prop(root_md,'charge')

	if weights == 'charge':
		weights = charge
	elif weights == None:
		weights = None

	# cluster the trajectory and output the index of each cluster
	clusters = cluster_traj(prefix,smile,save_super=True)
	indexes = []
	for i in clusters:
		index = get_class_center(prefix,smile,i)
		indexes.append(index)

	#print('INDEXES OF SELECTED PDB {}'.format(indexes))

	prob = boltz(root_md,indexes,energy_key='v')
	#print('THE PROB IS {}'.format(prob))
	
	if max_out:
		pmax = np.where(prob==np.max(prob))
		#print('INDEX OF MAX PROP IS {}'.format(pmax))
	# enumerate all the cluster and prob
	ave_fps_G1 = 0
	ave_fps_G2 = 0

	ave_xyz = 0

	for (i_cluster,p) in zip(indexes,prob):
		#print('INDEX OF CLUSTER IS {}'.format(indexes.index(i_cluster)))
		pdbname = prefix+smile+'/'+str(i_cluster)+'.pdb'
		xyz, elems, elems_idx = pdbinfo(pdbname)

		rad = params['G1']
		ang = params['G2']

		n_feats_G1 = len(rad['eta'])
		n_feats_G2 = len(ang['lamb'])*len(ang['eta'])

		fps_G1 = np.zeros((len(elems),n_feats_G1),dtype=np.float32)
		fps_G2 = np.zeros((len(elems),n_feats_G2),dtype=np.float32)

		
		for i,eta in enumerate(rad['eta']):
			datas = sum_G1(xyz,eta=eta,Rs=rad['Rs'][i],weights=weights)
			for j,data in enumerate(datas):
				fps_G1[j][i]=data

		feature = 0
		for i,lamb in enumerate(ang['lamb']):
			for j,eta in enumerate(ang['eta']):
				datas = sum_G2(xyz,lamb=lamb,eta=eta,xi=ang['xi'][j],weights=weights)
				feature += 1
				for k,data in enumerate(datas):
					fps_G2[k][feature-1] = data

		ave_fps_G1 += p*fps_G1
		ave_fps_G2 += p*fps_G2

		ave_xyz += p*xyz

		if np.where(np.array(indexes)==i_cluster) == pmax:
			#print('CATCH YOU!')
			elemsidx = np.array(elems_idx, np.int32)
			coord = np.array(xyz,np.float32)
			fps = np.concatenate((fps_G1, fps_G2),axis=-1)
			logp = float(logp)
	
			dataset['max_elems'] = elemsidx
			dataset['max_coord'] = coord
			dataset['max_fps']	= fps
			dataset['max_logp'] = logp



	elemsidx = np.array(elems_idx, np.int32)
	coord = np.array(ave_xyz,np.float32)
	fps = np.concatenate((ave_fps_G1,ave_fps_G2),axis=-1)
	logp = float(logp)

	dataset['elems'] = elemsidx
	dataset['coord'] = coord
	dataset['fps'] = fps
	dataset['logp'] = logp

	return dataset

if __name__ == '__main__':
	params = {'G1':{'eta':np.arange(1,1.8,0.1),'Rs':np.arange(0,0.8,0.1)},
          'G2':{'lamb':[1.0,-1.0],'eta':np.arange(1,1.8,0.1),'xi':np.arange(1,17.,2)},}
	p,p_max = generator_feat_md('logp/DATASETS/MD/Martel/','Cc1cc2c(cc1C)NC(=O)C[C@H]2c3ccccc3OC',4.17,params,weighs=None,max_out=True)
