#!/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from rdkit import Chem

def load_desc(data,elems_specified):
	smiles = data['SMILES'].values
	total = len(smiles)
	idx = []
	count = 0
	for i,smile in enumerate(smiles):
		m = Chem.MolFromSmiles(smile)
		if m is not None:
			atomlist = [atom.GetSymbol() for atom in m.GetAtoms()]

			if False in [atom in elems_specified for atom in atomlist]:
				count = count + 1
			else:
				idx.append(i)
		else:
			print('SMILE is wrong!')
	print('{}/{} is out of elements we specified'.format(count,total))
	return data.iloc[idx]
