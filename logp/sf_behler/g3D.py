#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
# 导入rdkit的绘图package
from rdkit.Chem import Draw


def SMILEto3D(mol):
	#m = Chem.MolFromSmiles(smile)
	m3d = Chem.AddHs(mol)
	return AllChem.EmbedMolecule(m3d, randomSeed=1)

def XYZ_dict(mol,method='MMFF'):
	"""
	method = MMFF (default) / UFF
	return the coordinate and the elements
	"""
	coords = []
	elem_idx = []
	elements = []

	mH = Chem.AddHs(mol)
	n = mH.GetNumAtoms()

	# Initial guess coordinate with Embed method
	#AllChem.EmbedMolecule(mH,useRandomCoords=True, randomSeed=1)
	AllChem.EmbedMolecule(mH,useRandomCoords=True)

	# Choose
	if method == 'MMFF':
		AllChem.MMFFOptimizeMolecule(mH)
	elif method == 'UFF':
		AllChem.UFFOptimizeMolecule(mH)	
	else:
		raise Exception('Method %s not Embedded! Please set the method to MMFF or UFF' %(method))

	for i in range(n):
		#element = mH.GetAtomWithIdx(i).GetSymbol()
		pos = mH.GetConformer().GetAtomPosition(i)
		coords.append([ pos.x, pos.y, pos.z])

	#for i in range(n):
	#	elem_idx.append(mH.GetAtomWithIdx(i))
	for atom in mH.GetAtoms():
		elem_idx.append(atom.GetAtomicNum())	

	for i in range(n):
		elements.append(mH.GetAtomWithIdx(i).GetSymbol())
	
	#print(coords)
	return np.array(coords),np.array(elements),np.array(elem_idx)


if __name__ == '__main__':
	smile = 'c1ccccc1'
	mol = Chem.MolFromSmiles(smile)
	#print(SMILEto3D(smile))
	xyz,elements= XYZ_dict(mol,method='UFF')
	print(xyz)
	print(elements)
