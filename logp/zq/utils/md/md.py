#!/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from openforcefield.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmmforcefields.generators import GAFFTemplateGenerator
from simtk.openmm.app import ForceField

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

from mdtraj.reporters import HDF5Reporter

from logp.zq.utils.g3D.g3D import smi2pdb


###
#  Here, we trimmed the symbol '/' from smiles just for the
#  construction of the direction.

def md(smile,params=None):
	"""
	args
	- smile: 
		SMILE format for a single molecule
	- params: a dictionary contain
		temp: temperature in the unit of Kevin
		pdbfreq: snapshot output frequency
		outfrep: frequency of information printed onto the screen
		steps: total steps 
		nonbondedMethod: method utilized for nonbonded interactions
		cutoff: cutoff distance
		constraints: strategy for constraints
		method: methods utilized to generate force field for small molecule
			'gaff' or 'smirnoff' is possible
		dir: path for storing the parameters and outputs
		'DumpFF': whether to dump the ff parameters
		'HasFF': whether we have generated the ff parameters
	"""
	if params is None:
		params = {'temp':300,'pdbfreq':1000,'outfreq':1000,'steps':10000,\
					'nonbondedMethod':PME,'cutoff':1.0*nanometer,'constraints':HBonds,\
					'method':'gaff','dir':'./', 'DumpFF':True,'HasFF':True,'SaveVel':False}

	prefix_name = params['dir']+str(smile).replace('/','')
	#print(prefix_name)

	if params['HasFF'] == False: # if you don't have force field please start from stractch!!!
		# create an openforcefield molecule object for molecule SMILEs
		molecule = Molecule.from_smiles(smile,allow_undefined_stereo=True)
		# Create the SMIRNOFF template generator with the most up to date Open Force Field Initiative force field
		if params['method'] == 'gaff':
			gaff = GAFFTemplateGenerator(molecules=molecule)
			# Create an OpenMM ForceField object with AMBER ff14SB and TIP3P with compatible ions
			forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
			# Register the SMIRNOFF template generator
			forcefield.registerTemplateGenerator(gaff.generator)
		elif params['method'] == 'smirnoff':
			smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
			# Create an OpenMM ForceField object with AMBER ff14SB and TIP3P with compatible ions
			forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
			# Register the SMIRNOFF template generator
			forcefield.registerTemplateGenerator(smirnoff.generator)
		# Section to convert SMI to PDB
		smi2pdb(smile,prefix_name)
		# Section to load pdb file
		pdb = PDBFile(prefix_name+'/'+str(smile).replace('/','')+'.pdb')
		system = forcefield.createSystem(pdb.topology, nonbondedMethod=params['nonbondedMethod'],\
							nonbondedCutoff=params['cutoff'], constraints=params['constraints'])

		# dump force file into XML file format
		if params['DumpFF'] == True:
			ffSerial = XmlSerializer.serialize(system)
			print(ffSerial, file=open(prefix_name+'/output.xml','w+'))
	
	# has ff generated, so just load it !!!
	else:
		pdb = PDBFile(prefix_name+'/'+str(smile)+'.pdb')
		tmp_system = open(prefix_name+'/output.xml').read()
		system = XmlSerializer.deserialize(tmp_system)
	
	# get the nonbonded parameters for atoms
	chargelist = getChargeVectors(system)
	sigmalist = getSigmaVectors(system)
	epsilonlist = getEpsilonVectors(system)

	# dump all
	dumpVectors(chargelist,'charge',prefix_name+'/output')
	dumpVectors(sigmalist,'sigma',prefix_name+'/output')
	dumpVectors(epsilonlist,'epsi',prefix_name+'/output')

	# integrator method
	integrator = LangevinMiddleIntegrator(params['temp']*kelvin, 1/picosecond, 0.002*picoseconds)
	# simulation setup
	simulation = Simulation(pdb.topology,system,integrator)
	simulation.context.setPositions(pdb.positions)
	#forces = simulation.context.getState(getVelocities=True, getForces=True).getForces(asNumpy=True)
	simulation.minimizeEnergy()
	simulation.reporters.append(PDBReporter(params['dir']+str(smile).replace('/','')+'/'+'output.pdb',params['pdbfreq']))
	# dump the reporter to output.log file 
	# information in the following order:
	# - step: whether to write the current step index
	# - potentialEnergy: whether to write the potential energy
	# - kineticEnergy: whether to write the kinetic energy
	# - totalEnergy: whether to write the total energy
	# - temperature: whether to write the total energy
	simulation.reporters.append(StateDataReporter(params['dir']+str(smile).replace('/','')+'/output.log',params['outfreq'],step=True,\
								potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True))

	if params['SaveVel']:
		h5_reporter = HDF5Reporter(params['dir']+str(smile).replace('/','')+'/output.traj.h5',params['outfreq'],coordinates=True, \
									time=True,cell=True, potentialEnergy=True, kineticEnergy=True,temperature=True, \
									velocities=True)
		#simulation.saveState(params['dir']+str(smile)+'/outputlast.state.xml')
		simulation.reporters.append(h5_reporter)

	simulation.step(params['steps'])

	simulation.saveState(params['dir']+str(smile).replace('/','')+'/outputlast.state.xml')


def md_vel_spectrum(smile,params=None):
	# restart from the 1 ns trajectory
	prefix_name = params['dir']+str(smile).replace('/','')

	pdb = PDBFile(prefix_name+'/'+str(smile).replace('/','')+'.pdb')

	# load the force field
	tmp_system = open(prefix_name+'/output.xml').read()
	system = XmlSerializer.deserialize(tmp_system)

	# integrator method
	integrator = LangevinMiddleIntegrator(params['temp']*kelvin, 1/picosecond, 0.002*picoseconds)
	# simulation setup
	simulation = Simulation(pdb.topology,system,integrator)
	# load the last state of the 1 ns trajectory | xyz | vel
	simulation.loadState(prefix_name+'/outputlast.state.xml')

	simulation.reporters.append(PDBReporter(prefix_name+'/output.spectrum.pdb',params['pdbfreq']))
	simulation.reporters.append(StateDataReporter(prefix_name+'/output.spectrum.log',params['outfreq'],step=True,\
								potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True))

	h5_reporter = HDF5Reporter(prefix_name+'/output.traj.spectrum.h5',params['outfreq'],coordinates=True, \
								time=True,cell=True, potentialEnergy=True, kineticEnergy=True,temperature=True, \
								velocities=True)
	simulation.reporters.append(h5_reporter)
	
	simulation.step(params['steps'])
	


#def getChargeVector(force):
#	""" Get partial charges for all particles"""
#	chargelist = []
#	Zndx = -1 # index in particle parameters where to find charge
#	if isinstance(force, NonbondedForce):
#		Zndx = 0
#	if isinstance(force, CustomNonbondedForce):
#		for i in range(len(force.getParticleParameters(0))):
#			if force.getPerParticleParameterName(i) == 'charge':
#				Zndx=i
#	if Zndx >=0:
#		chargelist = []
#		for i in range(force.getNumparticles()):
#			charge = force.getParticleParameters(i)[Zndx]
#			if isinstance(charge, Quantity):
#				charge = charge/ elementary_charge
#			chargelist.append(charge)
#		return chargelist
#	return None

def getChargeVectors(system):
	"""
	Get partial charges for all particles
	"""
	NbForce = system.getForces()[3]
	Numatoms = system.getNumParticles()
	chargelist = [NbForce.getParticleParameters(atom)[0]/elementary_charge for atom in range(Numatoms)]
	return chargelist

def getSigmaVectors(system):
	"""
	Get partial Sigma for all particles
	"""
	NbForce = system.getForces()[3]
	Numatoms = system.getNumParticles()
	sigmalist = [NbForce.getParticleParameters(atom)[1]/nanometer for atom in range(Numatoms)]
	return sigmalist

def getEpsilonVectors(system):
	"""
	Get partial Epsilon for all particles
	"""
	NbForce = system.getForces()[3]
	Numatoms = system.getNumParticles()
	epsilonlist = [NbForce.getParticleParameters(atom)[2]/kilojoule_per_mole for atom in range(Numatoms)]
	return epsilonlist

def dumpVectors(Vectors,key,filename):
	"""
	dump Vectors derived from `getChargeVectors | getSigmaVectors | getEpsilonVectors`
	to conrresponding file with postfix '.q','.sigm', and '.epsi'
	"""
	if key == 'charge':
		with open(filename+'.q','w') as f:
			f.write('# in the unit of elementary charge e\n')
			for i in Vectors:
				f.write(str(i)+'\n')

	if key == 'sigma':
		with open(filename+'.sigm','w') as f:
			f.write('# in the unit of nanometer\n')
			for i in Vectors:
				f.write(str(i)+'\n')

	if key == 'epsi':
		with open(filename+'.epsi','w') as f:
			f.write('# in the unit of kj/mol\n')
			for i in Vectors:
				f.write(str(i)+'\n')
				



if __name__ == '__main__':
	smile = 'c1ccccc1'
	params = {'temp':300,'pdbfreq':1000,'outfreq':1000,'steps':10000,\
						'nonbondedMethod':NoCutoff,'cutoff':1.0*nanometer,'constraints':HBonds,\
						'method':'gaff',\
						'dir':'logp/DATASETS/MD/'}
	
	md(smile,params)
