#!/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from logp.zq.io.base import list_loader
from logp.sf_behler.g3D import XYZ_dict
from logp.sf_behler.sf import sum_G1, sum_G2
from rdkit import Chem
from logp.sf_behler.sf import get_charge


def load_feat(datalist,**kwargs):
	"""convert smiles to features
	"""
	
	@list_loader(feat_num=4)
	def _feat_gen(data):
		smile = data[0]
		label = data[1]
		
		mol = Chem.MolFromSmiles(smile)
		xyz, elems, elems_idx = XYZ_dict(mol, method='UFF')
		G1_1 = sum_G1(xyz,eta=2,Rs=0)
		G1_2 = sum_G1(xyz,eta=3,Rs=1)
		G2_1 = sum_G2(xyz,lamb=1,eta=2,xi=1)
		G2_2 = sum_G2(xyz,lamb=-1,eta=2,xi=1)
	
		G1_1 = np.array(G1_1).reshape(-1,1)
		G1_2 = np.array(G1_2).reshape(-1,1)
		G2_1 = np.array(G2_1).reshape(-1,1)
		G2_2 = np.array(G2_2).reshape(-1,1)

		elemsidx = np.array(elems_idx, np.int32)
		coord = np.array(xyz, np.float32)
		fps = np.concatenate((G1_1,G1_2,G2_1,G2_2),axis=1)
		logp = float(label)

		data = {'elems':elemsidx, 'coord':coord, 'fps':fps, 'logp':logp}
		return data
	
	return _feat_gen(datalist,**kwargs)

def load_feat_from_csv(datalist,feat_num,**kwargs):
	
	@list_loader(feat_num=feat_num)
	def _feat_gen(data):
		elems = data[0]
		coord = data[1]
		fps = data[2]
		logp = data[3]

		data = {'elems':elems,'coord':coord,'fps':fps,'logp':logp}
		return data
	
	return _feat_gen(datalist,**kwargs)

def generator(smiles,logps,params):
	dataset = {'elems':[],'coord':[],'fps':[],'logp':[]}
	for x,smile in enumerate(smiles):
		mol = Chem.MolFromSmiles(smile)
		mH = Chem.AddHs(mol)
		charge = get_charge(mH)
		xyz, elems, elems_idx = XYZ_dict(mol,method=params['method'])
		rad = params['G1']
		ang = params['G2']
		
		n_feats_G1 = len(rad['eta']) 
		n_feats_G2 = len(ang['lamb'])*len(ang['eta'])

		fps_G1 = np.zeros((len(elems),n_feats_G1),dtype=np.float32)
		fps_G2 = np.zeros((len(elems),n_feats_G2),dtype=np.float32)

		for i,eta in enumerate(rad['eta']):
			datas = sum_G1(xyz,eta=eta,Rs=rad['Rs'][i],weights=charge)
			#print(datas)
			for j,data in enumerate(datas):
				fps_G1[j][i]=data

		feature = 0
		for i,lamb in enumerate(ang['lamb']):
			for j,eta in enumerate(ang['eta']):
				datas = sum_G2(xyz,lamb=lamb,eta=eta,xi=ang['xi'][j],weights=charge)
				#print(datas)
				feature += 1
				for k,data in enumerate(datas):
					fps_G2[k][feature-1] = data

		elemsidx = np.array(elems_idx, np.int32)
		coord = np.array(xyz,np.float32)
		fps = np.concatenate((fps_G1,fps_G2),axis=-1)
		logp = float(logps[x])

		dataset['elems'].append(elemsidx)
		dataset['coord'].append(coord)
		dataset['fps'].append(fps)
		dataset['logp'].append(logp)
		
		if x%25 == 0:
			print("{} processed!".format(x))
		
	return dataset

def generator_feat(smile,logp,params,weights='charge'):
	dataset = {'elems':[],'coord':[],'fps':[],'logp':[]}
	mol = Chem.MolFromSmiles(smile)
	mH = Chem.AddHs(mol)
	charge = get_charge(mH)
	if weights == 'charge':
		weights = charge
	elif weights == None:
		weights = None

	xyz, elems, elems_idx = XYZ_dict(mol,method=params['method'])
	rad = params['G1']
	ang = params['G2']
	
	n_feats_G1 = len(rad['eta']) 
	n_feats_G2 = len(ang['lamb'])*len(ang['eta'])

	fps_G1 = np.zeros((len(elems),n_feats_G1),dtype=np.float32)
	fps_G2 = np.zeros((len(elems),n_feats_G2),dtype=np.float32)

	for i,eta in enumerate(rad['eta']):
		datas = sum_G1(xyz,eta=eta,Rs=rad['Rs'][i],weights=weights)
		#print(datas)
		for j,data in enumerate(datas):
			fps_G1[j][i]=data

	feature = 0
	for i,lamb in enumerate(ang['lamb']):
		for j,eta in enumerate(ang['eta']):
			datas = sum_G2(xyz,lamb=lamb,eta=eta,xi=ang['xi'][j],weights=weights)
			#print(datas)
			feature += 1
			for k,data in enumerate(datas):
				fps_G2[k][feature-1] = data

	elemsidx = np.array(elems_idx, np.int32)
	coord = np.array(xyz,np.float32)
	fps = np.concatenate((fps_G1,fps_G2),axis=-1)
	logp = float(logp)

	dataset['elems'] = elemsidx
	dataset['coord'] = coord
	dataset['fps'] = fps
	dataset['logp'] = logp
		
	return dataset
