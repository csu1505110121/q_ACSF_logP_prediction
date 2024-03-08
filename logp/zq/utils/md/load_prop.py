#!/bin/python
# -*- coding:utf-8 -*-

import numpy as np

def load_prop(prefix,key):
	if key == 'charge':
		path = prefix+'/output'+'.q'
	if key == 'sigma':
		path = prefix+'/output'+'.sigm'
	if key == 'epsi':
		path = prefix+'/output'+'.epsi'
	prop = []
	with open(path,'r') as f:
		while True:
			line = f.readline()
			if not line:
				break
			else:
				if '#' not in line:
					prop.append(float(line.strip()))
	return prop


def load_energy(prefix,key):
	"""
	output.log is in the order of:
		0    |        1        |       2       |      3      |      4
		step | potentialEnergy | kineticEnergy | totalEnergy | temperature
	# key parameters:
		- 'v': potential energy
		- 'k': kinetic energy
		- 'e': total energy
	"""
	out = []
	with open(prefix+'/output.log','r') as f:
		while True:
			line = f.readline()
			if not line:
				break
			else:
				if "#" not in line:
					data = line.split(',')
					out.append([int(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4])])
	out = np.array(out)
	if key == 'v':
		return out[:,1]
	elif key == 'k':
		return out[:,2]
	elif key == 'e':
		return out[:,3]
	elif key == 't':
		return out[:,4]
	else:
		print('ENERGY  OUT OF RANGE !')
		
