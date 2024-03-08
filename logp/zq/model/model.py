#!/bin/python
# -*- coding:utf-8 -*-

import logp.zq.network.network

import tensorflow as tf
import numpy as np

from logp.zq.utils.utils import pi_named

default_params = {
	'e_scale': 1.0, # scale for prediction
	'e_unit': 1.0, # output unit of energy during prediction
	# Loss function multipliers
	'e_loss_multiplier': 1.0,
	# Optimizer related
	'learning_rate':3e-4, # learning rate
	'use_norm_clip': True, # to deal with gradient explosion or vanishing
	'norm_clip'    : 0.01, # see tf.clip_by_global_norm
	'use_decay'    : True, # Exponential decay
	'decay_step'   : 10000,# every x steps
	'decay_rate'   : 0.999,# scale by x
}


def potential_model(params,**kwargs):
	"""shortcut for generating potential model from paramters
	Args:
		params(str or dict): paramter dictionary or the model_dir
		**kwargs: additional options for the estimator, e.g. config
	"""
	import os
	import yaml
	from tensorflow.python.lib.io.file_io import FileIO
	from datetime import datetime
	
	# Section of loading and storing the parameters
	if isinstance(params, str):
		model_dir = params
		assert tf.compat.v1.gfile.Exists('{}/params.yml'.format(model_dir)),\
			"Parameters files not found."
		# load parameters defined in `params.yml`
		with FileIO(os.path.join(model_dir, 'params.yml'), 'r') as f:
			params = yaml.load(f, Loader=yaml.Loader)
	else:
		model_dir = params['model_dir']
		yaml.Dumper.ignore_aliases = lambda *args: True
		to_write = yaml.dump(params)
		params_path = os.path.join(model_dir, 'params.yml')
		if not tf.compat.v1.gfile.IsDirectory(model_dir):
			tf.compat.v1.gfile.MakeDirs(model_dir)
		if tf.compat.v1.gfile.Exists(params_path):
			original = FileIO(params_path, 'r').read()
			if original != to_write:
				tf.compat.v1.gfile.Rename(params_path, params_path+'.' +
						datetime.now().strftime('%y%m%d%H%M'))
		FileIO(params_path, 'w').write(to_write)
	
	model = tf.estimator.Estimator(
		model_fn=_potential_model_fn, params=params,
		model_dir=model_dir, **kwargs)
	return model


def _potential_model_fn(features,labels,mode,params):
	"""Model function for neural network potentials
		Args:
			
	"""
	if isinstance(params['network'],str):
		network_fn = getattr(logp.zq.network.network, params['network'])

	network_params = params['network_params']
	model_params = default_params.copy()
	model_params.update(params['model_params'])
	# data predicted by Network
	pred = network_fn(features, **network_params)

	ind = features['ind_1'] # ind_1 => id of molecule for each atom
	nbatch = tf.reduce_max(ind)+1 # Num of molecules in a batch
	pred = tf.math.unsorted_segment_sum(pred, ind[:, 0], nbatch)

	if mode == tf.estimator.ModeKeys.TRAIN:
		n_trainable = np.sum([np.prod(v.shape)
							for v in tf.compat.v1.trainable_variables()])
		print("Total number of trainable variables: {}".format(n_trainable))

		# define the loss function
		loss, metrics = _get_loss(features, pred, model_params)
		_make_train_summary(metrics) # make a summary
		# define the operation for variable training
		train_op = _get_train_op(loss, model_params)
		return tf.estimator.EstimatorSpec(
				mode, loss=loss, train_op=train_op)

	if mode == tf.estimator.ModeKeys.EVAL:
		loss, metrics = _get_loss(features, pred, model_params)
		metrics = _make_eval_metrics(metrics)
		return tf.estimator.EstimatorSpec(
			mode, loss=loss, eval_metric_ops=metrics)

	if mode == tf.estimator.ModeKeys.PREDICT:
		pred = pred / model_params['e_scale']
		pred *= model_params['e_unit']
		
		predictions = {'logp':pred}
		
		return tf.estimator.EstimatorSpec(mode,predictions=predictions)

@pi_named('LOSSES')
def _get_loss(features, pred, model_params):
	metrics = {} # Not editting features here for safety, use a separate dict
	
	e_pred = pred
	e_data = features['logp']
	
	e_data *= model_params['e_scale']
	
	e_error = pred - e_data
	metrics['e_data'] = e_data
	metrics['e_pred'] = e_pred
	metrics['e_error'] = e_error

	#print(metrics['e_data'])
	#print(metrics['e_pred'])
	#print(metrics['e_error'])

	e_loss = e_error**2 * model_params['e_loss_multiplier']
	metrics['e_loss'] = e_loss
	tot_loss = tf.reduce_mean(e_loss)

	metrics['tot_loss'] = tot_loss
	return tot_loss, metrics

@pi_named('TRAIN_OP')
def _get_train_op(loss, model_params):
	# Get the optimizer
	global_step = tf.compat.v1.train.get_global_step()
	learning_rate = model_params['learning_rate']
	if model_params['use_decay']:
		# the function returns the decayed learning rate. 
		# It is computed as:
		# decayed_learning_rate = learning_rate *
		#   decay_rate ^(global_step/ decay_steps)
		learning_rate = tf.compat.v1.train.exponential_decay(
			learning_rate, global_step,
			model_params['decay_step'], model_params['decay_rate'],
			staircase=True)

	tf.compat.v1.summary.scalar('learning rate',learning_rate)
	# define the optimizer
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
	# Get the gradients
	tvar = tf.compat.v1.trainable_variables()
	grads = tf.gradients(loss, tvar)

	if model_params['use_norm_clip']:
		grads, _ = tf.clip_by_global_norm(grads, model_params['norm_clip'])
	
	return optimizer.apply_gradients(zip(grads,tvar), global_step = global_step)

@pi_named('METRICS')
def _make_train_summary(metrics):
	tf.compat.v1.summary.scalar('E_RMSE', tf.sqrt(tf.reduce_mean(metrics['e_error']**2)))
	tf.compat.v1.summary.scalar('E_MAE', tf.reduce_mean(tf.abs(metrics['e_error'])))
	tf.compat.v1.summary.scalar('E_LOSS', tf.reduce_mean(metrics['e_loss']))
	tf.compat.v1.summary.scalar('TOT_LOSS', metrics['tot_loss'])

	tf.compat.v1.summary.histogram('E_DATA', metrics['e_data'])
	tf.compat.v1.summary.histogram('E_PRED', metrics['e_pred'])
	tf.compat.v1.summary.histogram('E_ERROR', metrics['e_error'])

@pi_named('METRICS')
def _make_eval_metrics(metrics):
	eval_metrics = {
		'METRICS/E_MAE': tf.compat.v1.metrics.mean_absolute_error(
				metrics['e_data'],metrics['e_pred']),
		'METRICS/E_MSE': tf.compat.v1.metrics.mean_squared_error(
				metrics['e_data'],metrics['e_pred']),
		'METRICS/E_RMSE': tf.compat.v1.metrics.root_mean_squared_error(
				metrics['e_data'],metrics['e_pred']),
		'METRICS/E_LOSS': tf.compat.v1.metrics.mean(metrics['e_loss']),
		'METRICS/TOT_LOSS': tf.compat.v1.metrics.mean(metrics['tot_loss'])
	}
	
	return eval_metrics




