#!/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

def load_pd(filename):
	dataset = pd.read_csv(filename)
	
	#results = {'elems':[],'coord':[],'fps':[],'logp':[]}
	results = []
	
	#print(feat_len)
	for i in range(dataset.shape[0]):
		elems = [int(elem) for elem in dataset.iloc[i]['elems'].strip('[]').split()]
		elems = np.array(elems)

		n_atoms = len(elems)

		coord = [float(xyz) for xyz in dataset.iloc[i]['coord'].replace('[','').replace(']','').replace('\n','').split()]
		coord = np.array(coord).reshape(-1,3)

		fps = [float(fp) for fp in dataset.iloc[i]['fps'].replace('[','').replace(']','').replace('\n','').split()]
		fps = np.array(fps).reshape(n_atoms,-1)

		logp = float(dataset.iloc[i]['logp'])
		
		#print(elems)	
		#print(coord)
		#print(fps)

		#results['elems'].append(elems)
		#results['coord'].append(coord)
		#results['fps'].append(fps)
		#results['logp'].append(logp)
		results.append((elems,coord,fps,logp))
	return results



