#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def dump_pdb(filename,coord,elements):
	"""
	borrowed from my project DA2-MC
	"""
	with open(filename,'w') as f:
		f.write('CREATED BY QIANG \nDA2 SAMPLING REVISION!\n')
		for i in range(coord.shape[0]):
			f.write('%-6s%5d%1s%-4s%1s%-3s%1s%1s%4d%4s%8.3f%8.3f%8.3f%6.2f%6.2f%7s\n' \
					%('ATOM',i,'',elements[i],'','','','',1,'',\
						coord[i][0],coord[i][1],coord[i][2],\
						1.0,1.0,''))

def dump_xyz(filename,coord,elements):
	"""
	borrowed from my project Optimization
	"""
	with open(filename,'w') as f:
		f.write(str(coord.shape[0])+'\n')
		f.write('description: Created by Qiang'+'\n')
		for i in range(coord.shape[0]):
			f.write(elements[i]+'\t'+ str(coord[i][0]) + '\t' \
									+ str(coord[i][1]) + '\t' \
									+ str(coord[i][2]) + '\n')

if __name__ == '__main__':
	pass

