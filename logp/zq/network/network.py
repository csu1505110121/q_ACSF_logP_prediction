#!/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

def make_fps(tensors, nn_spec):
	fps = {e: [] for e in nn_spec.keys()}
	fps_all = tensors['fps']
	for e in nn_spec.keys():
		ind = tf.where(tf.equal(tensors['elems'],e))
		fps[e].append(tf.gather_nd(fps_all, ind))
	# Concatenate all fingerprints
	fps = {k: tf.concat(v,axis=-1) for k,v in fps.items()}
	return fps


def bpnn(tensors, nn_spec,act='tanh'):
	"""Network function for Behler-Parrinello Neural Network"""

	fps = make_fps(tensors, nn_spec)
	output = 0.0
	
	n_atoms = tf.shape(tensors['elems'])[0]
	for k, v in nn_spec.items():
		ind = tf.where(tf.equal(tensors['elems'],k))
		with tf.compat.v1.variable_scope("BP_DENSE_{}".format(k)):
			nodes = fps[k]
			for n_node in v:
				nodes = tf.compat.v1.layers.dense(nodes, n_node, activation=act)
			atomic_en = tf.compat.v1.layers.dense(nodes, 1, activation=None,
								use_bias=False, name='E_OUT_{}'.format(k))
		output += tf.math.unsorted_segment_sum(atomic_en[:,0],ind[:,0],n_atoms)
	return output
