#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import mdtraj as md

def mdrmsd(prefix,smile):
	filename = prefix + str(smile)+'/output.pdb'
	traj =  md.load(filename)
	# pick heavy atoms
	atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']
	
	rmsd = md.rmsd(traj,traj,frame=0,atom_indices=atom_indices)

	return rmsd



