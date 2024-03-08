#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sf import sum_G1, sum_G2
from dump import dump_pdb, dump_xyz
from g3D import XYZ_dict
import os

directory = 'optimized_pdb'

isExists = os.path.exists(directory)

if not isExists:
	os.mkdir(directory)
	print(directory+'is created successfully!')
else:
	print(directory+'Existed!')

# filename for the database
filename = '../descriptors.csv'

data = pd.read_csv(filename,index_col=[0])

smiles = data.index
logps = data.logp.values


symm_func = {'index':[],\
			'elements':[],\
			'G1':[],\
			'G2_positive':[],\
			'G2_negative':[],\
			'logp':[]}

for i in range(len(smiles)):
	try:
		smile = smiles[i]
		mol = Chem.MolFromSmiles(smile)
		xyz,elements= XYZ_dict(mol,method='UFF')
		filename = directory+'/'+smile+'.pdb'
		dump_pdb(filename,xyz,elements)
		symm_func['index'].append(i+1)
	
		symm_func['elements'].append(elements)
	
		sumG1 = sum_G1(xyz)
		symm_func['G1'].append(sumG1)
	
		sumG2_p = sum_G2(xyz,lamb=1)
		symm_func['G2_positive'].append(sumG2_p)
	
		sumG2_n = sum_G2(xyz,lamb=-1)
		symm_func['G2_negative'].append(sumG2_n)
		
		symm_func['logp'].append(logps[i])
	except:
		print('Faild for %s'%smiles[i])

results = pd.DataFrame(symm_func)

results.to_csv('sf_results.csv',index=False)
