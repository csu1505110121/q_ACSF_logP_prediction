#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from rdkit import Chem

def filt_elems(data,elems_specified):
	results = {'smiles':[],'logp':[]}
	smiles = data['SMILES'].values
	total = len(smiles)
	count = 0
	for i,smile in enumerate(smiles):
		m = Chem.MolFromSmiles(smile)
		if m is not None:
			atomlist = [atom.GetSymbol() for atom in m.GetAtoms()]
		
			if False in [atom in elems_specified for atom in atomlist]:
				#print('Out of elems {} specified'.format(elems_specified))
				count = count + 1
			else:
				results['smiles'].append(smile)
				results['logp'].append(data['logp'][i])
		else:
			print('Generated Failed!')
	print(len(results['smiles']))
	print('{}/{} is out of elements we specified'.format(count, total))
	return results

if __name__ == '__main__':
	elems_specified = ['C','H','O','N']
	dataset = '../DATASETS/DATASETS_TOTAL.xlsx'
	dataset_martel = pd.read_excel(dataset,sheet_name='Martel')
	#print(dataset_martel)
	dataset_filt = filt_elems(dataset_martel,elems_specified)
	pd = pd.DataFrame(data=dataset_filt)
	print(pd)
