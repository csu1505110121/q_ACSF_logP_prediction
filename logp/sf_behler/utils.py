#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from functools import wraps

# Codes here borrewed from pinn

def pi_named(default_name='unnamed'):
	"""decorate a layer to have a name"""
	def decorator(func):
		@wraps(func)
		def named_layer(*args, name=default_name, **kwargs):
			with tf.compat.v1.variable_scope(name):
				return func(*args, **kwargs)
		return named_layer
	return decorator


@pi_named('G2_symm_func')
def G2_SF(tensors, Rs, eta, i, j):
	""" Belher-Parrinello G2 symmetry functions. 
	Args:
		i: central atom type,
		j: adjacent atom type,
		Rs: a list of Rs values,
		eta: a list of eta values
		***Note***:
		Rs and eta should be the same length
	
	Returns:
		fp: a (n_atom * n_fingerprint) tensor of fingerprints
	"""

	R = tensors['dist']
	fc = tensors['cutoff_func']
	# Compute p_filter => boolean mask of relavent pairwise interactions
	p_filter = None
	a_filter = None
	# relative position of i atom in the "to_output" group (defined by i)
	i_rind = tensors['ind_2'][:, 0]
	a_rind = tf.cumsum(tf.ones_like(tensors['elems'], tf.int32))-1
	if i != 'ALL':
		i_elem = tf.gather(tensors['elems'], i_rind)
		p_filter = tf.math.equal(i_elem, i)
		a_rind = tf.math.cumsum(tf.cast(tf.equal(tensors['elems'], i), tf.int32))-1
	if j != 'ALL':
		j_elem = tf.gather(tensors['elems'], tensors['ind_2'][:, 1])
		j_filter = tf.math.equal(j_elem, j)
		p_filter = tf.math.reduce_all(
			[p_filter, j_filter], axis=0) if p_filter is not None else j_filter
	
	# Gather the interactions
	if p_filter is not None:
		p_ind = tf.cast(tf.where(p_filter)[:, 0], tf.int32)
		R = tf.gather(R, p_ind)
		fc = tf.gather(fc,p_ind)
		i_rind = tf.gather(a_rind, tf.gather(i_rind, p_ind))
	else:
		p_ind = tf.math.cumsum(tf.ones_like(i_rind))-1
	
	# Symmetry Functions
	n_sf = len(Rs)
	R = tf.expand_dims(R,1)
	fc = tf.expand_dims(fc, 1)
	Rs = tf.expand_dims(Rs, 0)
	eta = tf.expand_dims(eta, 0)
	sf = tf.math.exp(-eta*(R-Rs)**2)*fc
	fp = tf.scatter_nd(tf.expand_dims(i_rind,1),sf,
					[tf.math.reduce_max(a_rind)+1,n_sf])
	jacob = tf.stack([tf.gradients(sf[:, i], tensors['diff'])[0]
					for i in range(n_sf)], axis=2)
	jacob = tf.gather_nd(jacob, tf.expand_dims(p_ind, 1))
	jacob_ind = tf.stack([p_ind, i_rind], axis=1)
	return fp, jacob, jacob_ind

@pi_named('G3_symm_func')
def G3_SF(tensors, cutoff_type, rc, lambd, zeta, eta, i="ALL", j="ALL", k="ALL"):
	"""BP-style G3 symmetry functions.
	Args:
		cutoff_type, rc: cutoff function and radius
		lambd: a list of lambda values.
		zeta: a list of zeta values.
		eta: a list of eta values.
		i, j, k: atom types (as int32)

	Returns:
		fp: a (n_atom x n_fingerprint) tensor of fingerprints
			where n_atom is the number of central atoms defined by "i"
		jacob: a (n_pair x n_fingerprint) tensor 
			where n_pair is the number of relavent pairs in this SF
		jacob_ind: a (n_pair) tensor 
			each row correspond to the (p_ind, i_rind) of the pair
			p_ind => the relative position of this pair within all pairs
			i_rind => the index of the central atom for this pair
	"""
	
	if 'ind_3' not in tensors:
		tensors['ind_3'] = _form_triplet(tensors)
	
	R = tensors['dist']
	fc = tensors['cutoff_func']
	diff = tensors['diff']
	ind_ij = tensors['ind_3'][:, 0]
	ind_ik = tensors['ind_3'][:, 1]
	ind2 = tensors['ind_2']
	i_rind = tf.gather(tensors['ind_2'][:, 0], ind_ij)
	
	#Build triplet filter
	t_filter = None
	a_rind = tf.math.cumsum(tf.ones_like(tensors['elems'], tf.int32))-1
	if i != 'ALL':
		i_elem = tf.gather(tensors['elems'], tf.gather(ind2[:, 0], ind_ij))
		t_filter = tf.math.equal(i_elem, i)
		a_rind = tf.math.cumsum(tf.cast(tf.equal(tensors['elems'], i), tf.int32))-1
	if j != 'ALL':
		j_elem = tf.gather(tensors['elems'], tf.gather(ind2[:, 1], ind_ij))
		j_filter = tf.equal(j_elem, j)
		t_filter = tf.math.reduce_all(
				[t_filter, j_filter], axis=0) if t_filter is not None else j_filter
	if k != 'ALL':
		k_elem = tf.gather(tensors['elems'],tf.gather(ind2[:,1],ind_ik))
		k_filter = tf.math.equal(k_elem, k)
		t_filter = tf.math.reduce_all(
				[t_filter, k_filter], axis=0) if t_filter is not None else k_filter

	if t_filter is not None:
		t_ind = tf.where(t_filter)[:, 0]
		ind_ij = tf.gather(ind_ij, t_ind)
		ind_ik = tf.gather(ind_ik, t_ind)
		i_rind = tf.gather(a_rind, tf.gather(i_rind, t_ind))
	# Filter according to R_jk, once more
	diff_ij = tf.gather_nd(diff, tf.expand_dims(ind_ij, 1))
	diff_ik = tf.gather_nd(diff, tf.expand_dims(ind_ik, 1))
	diff_jk = diff_ik - diff_ij
	R_jk = tf.norm(diff_jk, axis=1)
	t_ind = tf.where(R_jk < rc)[:, 0]
	R_jk = tf.gather(R_jk, t_ind)
	fc_jk = cutoff_func(R_jk, cutoff_type=cutoff_type, rc=rc)
	ind_ij = tf.gather(ind_ij, t_ind)
	ind_ik = tf.gather(ind_ik, t_ind)
	i_rind = tf.gather(i_rind, t_ind)
	diff_ij = tf.gather_nd(diff_ij, tf.expand_dims(t_ind, 1))
	diff_ik = tf.gather_nd(diff_ik, tf.expand_dims(t_ind, 1))
	# G3 symmetry function
	R_ij = tf.expand_dims(tf.gather(R, ind_ij), 1)
	R_ik = tf.expand_dims(tf.gather(R, ind_ik), 1)
	R_jk = tf.expand_dims(R_jk, 1)
	fc_ij = tf.expand_dims(tf.gather(fc, ind_ij), 1)
	fc_ik = tf.expand_dims(tf.gather(fc, ind_ik), 1)
	fc_jk = tf.expand_dims(fc_jk, 1)
	diff_ij = tf.expand_dims(diff_ij, 1)
	diff_ik = tf.expand_dims(diff_ik, 1)
	eta = tf.expand_dims(eta, 0)
	zeta = tf.expand_dims(zeta, 0)
	lambd = tf.expand_dims(lambd, 0)
	# SF definition
	sf = 2**(1-zeta) *\
		(1+lambd*tf.math.reduce_sum(diff_ij*diff_ik, axis=-1)/R_ij/R_ik)**zeta *\
		tf.math.exp(-eta*(R_ij**2+R_ik**2+R_jk**2))*fc_ij*fc_ik*fc_jk
	fp = tf.scatter_nd(tf.expand_dims(i_rind, 1), sf,
					[tf.reduce_max(a_rind)+1, tf.shape(eta)[1]])
	# Generate Jacobian
	n_sf = sf.shape[-1]
	p_ind, p_uniq_idx = tf.unique(tf.concat([ind_ij, ind_ik], axis=0))
	i_rind = tf.math.unsorted_segment_max(
			tf.concat([i_rind, i_rind], axis=0), p_uniq_idx, tf.shape(p_ind)[0])
	jacob = tf.stack([tf.gradients(sf[:, i], tensors['diff'])[0]
								for i in range(n_sf)], axis=2)
	jacob = tf.gather_nd(jacob, tf.expand_dims(p_ind,1))
	jacob_ind = tf.stack([p_ind, i_rind], axis=1)
	return fp, jacob, jacob_ind


@pi_named('G4_symm_func')
def G4_SF(tensors, lambd, zeta, eta, i="ALL", j="ALL", k="ALL"):
	"""BP-style G4 symmetry functions.

	lambd, eta should have the same length,
	each element corresponds to a symmetry function.

	Args:
		lambd: a list of lambda values.
		zeta: a list of zeta values.
		eta: a list of eta values.
		i, j, k: atom types (as int32)

	Returns:
		fp: a (n_atom x n_fingerprint) tensor of fingerprints
			where n_atom is the number of central atoms defined by "i"
		jacob: a (n_pair x n_fingerprint) tensor 
			where n_pair is the number of relavent pairs in this SF
		jacob_ind: a (n_pair) tensor 
			each row correspond to the (p_ind, i_rind) of the pair
			p_ind => the relative position of this pair within all pairs
			i_rind => the index of the central atom for this pair
	"""
	if 'ind_3' not in tensors:
		tensors['ind_3'] = _form_triplet(tensors)

	R = tensors['dist']
	fc = tensors['cutoff_func']
	diff = tensors['diff']
	ind_ij = tensors['ind_3'][:, 0]
	ind_ik = tensors['ind_3'][:, 1]
	ind2 = tensors['ind_2']
	i_rind = tf.gather(tensors['ind_2'][:, 0], ind_ij)
	# Build triplet filter
	t_filter = None
	a_rind = tf.math.cumsum(tf.ones_like(tensors['elems'], tf.int32))-1
	if i != 'ALL':
		i_elem = tf.gather(tensors['elems'], tf.gather(ind2[:, 0], ind_ij))
		t_filter = tf.math.equal(i_elem, i)
		a_rind = tf.math.cumsum(tf.cast(tf.math.equal(tensors['elems'], i), tf.int32))-1
	if j != 'ALL':
		j_elem = tf.gather(tensors['elems'], tf.gather(ind2[:, 1], ind_ij))
		j_filter = tf.math.equal(j_elem, j)
		t_filter = tf.math.reduce_all(
			[t_filter, j_filter], axis=0) if t_filter is not None else j_filter
	if k != 'ALL':
		k_elem = tf.gather(tensors['elems'], tf.gather(ind2[:, 1], ind_ik))
		k_filter = tf.math.equal(k_elem, k)
		t_filter = tf.math.reduce_all(
			[t_filter, k_filter], axis=0) if t_filter is not None else k_filter
	if t_filter is not None:
		t_ind = tf.where(t_filter)[:, 0]
		ind_ij = tf.gather(ind_ij, t_ind)
		ind_ik = tf.gather(ind_ik, t_ind)
		i_rind = tf.gather(a_rind, tf.gather(i_rind, t_ind))
	# G4 symmetry function
	R_ij = tf.expand_dims(tf.gather(R, ind_ij), 1)
	R_ik = tf.expand_dims(tf.gather(R, ind_ik), 1)
	fc_ij = tf.expand_dims(tf.gather(fc, ind_ij), 1)
	fc_ik = tf.expand_dims(tf.gather(fc, ind_ik), 1)
	diff_ij = tf.expand_dims(tf.gather_nd(diff, tf.expand_dims(ind_ij, 1)), 1)
	diff_ik = tf.expand_dims(tf.gather_nd(diff, tf.expand_dims(ind_ik, 1)), 1)
	eta = tf.expand_dims(eta, 0)
	zeta = tf.expand_dims(zeta, 0)
	lambd = tf.expand_dims(lambd, 0)
	sf = 2**(1-zeta) *\
		(1+lambd*tf.math.reduce_sum(diff_ij*diff_ik, axis=-1)/R_ij/R_ik)**zeta *\
		tf.exp(-eta*(R_ij**2+R_ik**2))*fc_ij*fc_ik
	fp = tf.scatter_nd(tf.expand_dims(i_rind, 1), sf,
					   [tf.math.reduce_max(a_rind)+1, tf.shape(eta)[1]])

	# Jacobian generation (perhaps needs some clarification)
	# In short, gradients(sf, diff) gives the non-zero parts of the
	# diff => sf Jacobian (Natom x Natom x 3)

	n_sf = sf.shape[-1]
	p_ind, p_uniq_idx = tf.unique(tf.concat([ind_ij, ind_ik], axis=0))
	i_rind = tf.math.unsorted_segment_max(
		tf.concat([i_rind, i_rind], axis=0), p_uniq_idx, tf.shape(p_ind)[0])
	jacob = tf.stack([tf.gradients(sf[:, i], tensors['diff'])[0]
					  for i in range(n_sf)], axis=2)
	jacob = tf.gather_nd(jacob, tf.expand_dims(p_ind, 1))
	jacob_ind = tf.stack([p_ind, i_rind], axis=1)
	return fp, jacob, jacob_ind


@pi_named('form_tripet')
def _form_triplet(tensors):
	"""Returns triplet indices [ij, jk], where r_ij, r_jk < r_c"""
	p_iind = tensors['ind_2'][:, 0]
	n_atoms = tf.shape(tensors['ind_1'])[0]
	n_pairs = tf.shape(tensors['ind_2'])[0]
	p_aind = tf.math.cumsum(tf.ones(n_pairs, tf.int32))
	p_rind = p_aind - tf.gather(tf.math.segment_min(p_aind, p_iind), p_iind)
	t_dense = tf.scatter_nd(tf.stack([p_iind, p_rind], axis=1), p_aind,
							[n_atoms, tf.reduce_max(p_rind)+1])
	t_dense = tf.gather(t_dense, p_iind)
	t_index = tf.cast(tf.where(t_dense), tf.int32)
	t_ijind = t_index[:, 0]
	t_ikind = tf.gather_nd(t_dense, t_index)-1
	t_ind = tf.gather_nd(tf.stack([t_ijind, t_ikind], axis=1),
						 tf.where(tf.not_equal(t_ijind, t_ikind)))
	return t_ind


@pi_named('bp_symm_func')
def bp_symm_func(tensors, sf_spec, rc, cutoff_type):
	""" Wrapper for building Behler-style symmetry functions"""
	sf_func = {'G2': G2_SF, 'G3': G3_SF, 'G4': G4_SF}
	fps = {}
	for i, sf in enumerate(sf_spec):
		options = {k: v for k, v in sf.items() if k != "type"}
		if sf['type'] == 'G3':  # Workaround for G3 only
			options.update({'rc': rc, 'cutoff_type': cutoff_type})
		fp, jacob, jacob_ind = sf_func[sf['type']](
			tensors,  **options)
		fps['fp_{}'.format(i)] = fp
		fps['jacob_{}'.format(i)] = jacob
		fps['jacob_ind_{}'.format(i)] = jacob_ind
	return fps


@tf.custom_gradient
def _fake_fp(diff, fp, jacob, jacob_ind, n_pairs):
	def _grad(dfp, jacob, jacob_ind):
		# Expand dfp to => (n_pairs x 3 x n_fp)
		dfp = tf.expand_dims(tf.gather_nd(dfp, jacob_ind[:, 1:]), axis=1)
		ddiff = tf.math.reduce_sum(jacob*dfp, axis=2)
		ddiff = tf.IndexedSlices(ddiff, jacob_ind[:, 0], [n_pairs, 3])
		return ddiff, None, None, None, None
	return tf.identity(fp), lambda dfp: _grad(dfp, jacob, jacob_ind)


def make_fps(tensors, sf_spec, nn_spec, use_jacobian, fp_range, fp_scale):
	fps = {e: [] for e in nn_spec.keys()}
	fps['ALL'] = []
	n_pairs = tf.shape(tensors['diff'])[0]
	for i, sf in enumerate(sf_spec):
		fp = tensors['fp_{}'.format(i)]
		if use_jacobian:
			# connect the diff -> fingerprint gradient
			fp = _fake_fp(tensors['diff'], fp,
						  tensors['jacob_{}'.format(i)],
						  tensors['jacob_ind_{}'.format(i)],
						  n_pairs)
		if fp_scale:
			fp = (fp-fp_range[i][0])/(fp_range[i][1]-fp_range[i][0])*2-1
		fps[sf['i']].append(fp)
	# Deal with "ALL" fingerprints
	fps_all = fps.pop('ALL')
	if fps_all != []:
		fps_all = tf.concat(fps_all, axis=-1)
		for e in nn_spec.keys():
			ind = tf.where(tf.equal(tensors['elems'], e))
			fps[e].append(tf.gather_nd(fps_all, ind))
	# Concatenate all fingerprints
	fps = {k: tf.concat(v, axis=-1) for k, v in fps.items()}
	return fps

@pi_named('cell_list_nl')
def cell_list_nl(tensors, rc=5.0):
	""" Compute neighbour list with celllist approach
	https://en.wikipedia.org/wiki/Cell_lists
	This is very lengthy and confusing implementation of cell list nl.
	Probably needs optimization outside Tensorflow.

	The function expects a dictionary of tensors from a sparse_batch
	with keys: 'ind_1', 'coord' and optionally 'cell'
	"""
	atom_sind = tensors['ind_1']
	atom_apos = tensors['coord']
	atom_gind = tf.math.cumsum(tf.ones_like(atom_sind), 0)
	atom_aind = atom_gind - 1
	to_collect = atom_aind
	if 'cell' in tensors:
		coord_wrap = _wrap_coord(tensors)
		atom_apos =  coord_wrap
		rep_apos, rep_sind, rep_aind = _pbc_repeat(
					coord_wrap, tensors['cell'], tensors['ind_1'], rc)
		atom_sind = tf.concat([atom_sind, rep_sind], 0)
		atom_apos = tf.concat([atom_apos, rep_apos], 0)
		atom_aind = tf.concat([atom_aind, rep_aind], 0)
		atom_gind = tf.math.cumsum(tf.ones_like(atom_sind), 0)
	atom_apos = atom_apos - tf.math.reduce_min(atom_apos, axis=0)
	atom_cpos = tf.concat(
			[atom_sind, tf.cast(atom_apos//rc, tf.int32)], axis=1)
	cpos_shap = tf.concat([tf.math.reduce_max(atom_cpos, axis=0) + 1, [1]], axis=0)
	samp_ccnt = tf.squeeze(tf.scatter_nd(
			atom_cpos, tf.ones_like(atom_sind, tf.int32), cpos_shap), axis=-1)
	cell_cpos = tf.cast(tf.where(samp_ccnt), tf.int32)
	cell_cind = tf.math.cumsum(tf.ones(tf.shape(cell_cpos)[0], tf.int32))
	cell_cind = tf.expand_dims(cell_cind, 1)
	samp_cind = tf.squeeze(tf.scatter_nd(
			cell_cpos, cell_cind, cpos_shap), axis=-1)
	# Get the atom's relative index(rind) and position(rpos) in cell
	# And each cell's atom list (alst)
	atom_cind = tf.gather_nd(samp_cind, atom_cpos) - 1
	atom_cind_args = tf.contrib.framework.argsort(atom_cind, axis=0)
	atom_cind_sort = tf.gather(atom_cind, atom_cind_args)

	atom_rind_sort = tf.math.cumsum(tf.ones_like(atom_cind, tf.int32))
	cell_rind_min = tf.math.segment_min(atom_rind_sort, atom_cind_sort)
	atom_rind_sort = atom_rind_sort - tf.gather(cell_rind_min, atom_cind_sort)
	atom_rpos_sort = tf.stack([atom_cind_sort, atom_rind_sort], axis=1)
	atom_rpos = tf.math.unsorted_segment_sum(atom_rpos_sort, atom_cind_args,
									tf.shape(atom_gind)[0])
	cell_alst_shap = [tf.shape(cell_cind)[0], tf.math.reduce_max(samp_ccnt), 1]
	cell_alst = tf.squeeze(tf.scatter_nd(
				atom_rpos, atom_gind, cell_alst_shap), axis=-1)
	# Get cell's linked cell list, for cells in to_collect only
	disp_mat = np.zeros([3, 3, 3, 4], np.int32)
	disp_mat[:, :, :, 1] = np.reshape([-1, 0, 1], (3, 1, 1))
	disp_mat[:, :, :, 2] = np.reshape([-1, 0, 1], (1, 3, 1))
	disp_mat[:, :, :, 3] = np.reshape([-1, 0, 1], (1, 1, 3))
	disp_mat = np.reshape(disp_mat, (1, 27, 4))
	cell_npos = tf.expand_dims(cell_cpos, 1) + disp_mat
	npos_mask = tf.math.reduce_all(
			(cell_npos >= 0) & (cell_npos < cpos_shap[:-1]), 2)
	cell_nind = tf.squeeze(tf.scatter_nd(
			tf.cast(tf.where(npos_mask), tf.int32),
			tf.expand_dims(tf.gather_nd(
				samp_cind, tf.boolean_mask(cell_npos, npos_mask)), 1),
			tf.concat([tf.shape(cell_npos)[:-1], [1]], 0)), -1)
	# Finally, a sparse list of atom pairs
	coll_nind = tf.gather(cell_nind, tf.gather_nd(atom_cind, to_collect))
	pair_ic = tf.cast(tf.where(coll_nind), tf.int32)
	pair_ic_i = pair_ic[:, 0]
	pair_ic_c = tf.gather_nd(coll_nind, pair_ic) - 1
	pair_ic_alst = tf.gather(cell_alst, pair_ic_c)

	pair_ij = tf.cast(tf.where(pair_ic_alst), tf.int32)
	pair_ij_i = tf.gather(pair_ic_i, pair_ij[:, 0])
	pair_ij_j = tf.gather_nd(pair_ic_alst, pair_ij) - 1

	diff = tf.gather(atom_apos, pair_ij_j) - tf.gather(atom_apos, pair_ij_i)
	dist = tf.norm(diff, axis=-1)
	ind_rc = tf.where((dist < rc) & (dist > 0))
	dist = tf.gather_nd(dist, ind_rc)
	diff = tf.gather_nd(diff, ind_rc)
	pair_i_aind = tf.gather_nd(tf.gather(atom_aind, pair_ij_i), ind_rc)
	pair_j_aind = tf.gather_nd(tf.gather(atom_aind, pair_ij_j), ind_rc)

	output = {
		'ind_2': tf.concat([pair_i_aind, pair_j_aind], 1),
		'dist': dist,
		'diff': diff
	}
	return output

def _displace_matrix(max_repeat):
	"""This is a helper function for cell_list_nl"""
	d = []
	n_repeat = max_repeat*2 + 1
	tot_repeat = tf.math.reduce_prod(n_repeat)
	for i in range(3):
		d.append(tf.math.cumsum(tf.ones(n_repeat, tf.int32), axis=i)
				 - max_repeat[i] - 1)
	d = tf.reshape(tf.stack(d, axis=-1), [tot_repeat, 3])
	d = tf.concat([d[:tot_repeat//2], d[tot_repeat//2+1:]], 0)
	return d


def _pbc_repeat(coord, cell, ind_1, rc):
	"""This is a helper function for cell_list_nl"""
	n_repeat = rc * tf.norm(tf.matrix_inverse(cell), axis=1)
	n_repeat = tf.cast(tf.ceil(n_repeat), tf.int32)
	max_repeat = tf.math.reduce_max(n_repeat, axis=0)
	disp_mat = _displace_matrix(max_repeat)

	repeat_mask = tf.math.reduce_all(
		tf.expand_dims(n_repeat, 1) >= tf.math.abs(disp_mat), axis=2)
	atom_mask = tf.gather(repeat_mask, ind_1)
	repeat_ar = tf.cast(tf.where(atom_mask), tf.int32)
	repeat_a = repeat_ar[:, :1]
	repeat_r = repeat_ar[:, 2]
	repeat_s = tf.gather_nd(ind_1, repeat_a)
	repeat_pos = (tf.gather_nd(coord, repeat_a) +
				  tf.reduce_sum(
					  tf.gather_nd(cell, repeat_s) *
					  tf.gather(tf.cast(tf.expand_dims(disp_mat, 2),
										tf.float32), repeat_r), 1))
	return repeat_pos, repeat_s, repeat_a

def _wrap_coord(tensors):
	"""wrap positions to unit cell"""
	cell = tf.gather_nd(tensors['cell'], tensors['ind_1'])
	coord = tf.expand_dims(tensors['coord'], -1)
	frac_coord = tf.linalg.solve(tf.transpose(cell, perm=[0, 2, 1]), coord)
	frac_coord %= 1
	coord = tf.matmul(tf.transpose(cell, perm=[0, 2, 1]), frac_coord)
	return tf.squeeze(coord, -1)

def connect_dist_grad(tensors):
	"""This function assumes tensors is a dictionary containing 'ind_2',
	'diff' and 'dist' from a neighbor list layer It rewirtes the
	'dist' and 'dist' tensor so that their gradients are properly
	propogated during force calculations
	"""
	tensors['diff'] = _connect_diff_grad(tensors['coord'], tensors['diff'],
										 tensors['ind_2'])
	if 'dist' in tensors:
		# dist can be deleted if the jacobian is cached, so we may skip this
		tensors['dist'] = _connect_dist_grad(tensors['diff'], tensors['dist'])


@tf.custom_gradient
def _connect_diff_grad(coord, diff, ind):
	"""Returns a new diff with its gradients connected to coord"""
	def _grad(ddiff, coord, diff, ind):
		natoms = tf.shape(coord)[0]
		if type(ddiff) == tf.IndexedSlices:
			# handle sparse gradient inputs
			ind = tf.gather_nd(ind, tf.expand_dims(ddiff.indices, 1))
			ddiff = ddiff.values
		dcoord = tf.math.unsorted_segment_sum(ddiff, ind[:, 1], natoms)
		dcoord -= tf.math.unsorted_segment_sum(ddiff, ind[:, 0], natoms)
		return dcoord, None, None
	return tf.identity(diff), lambda ddiff: _grad(ddiff, coord, diff, ind)


@tf.custom_gradient
def _connect_dist_grad(diff, dist):
	"""Returns a new dist with its gradients connected to diff"""
	def _grad(ddist, diff, dist):
		return tf.expand_dims(ddist/dist, 1)*diff, None
	return tf.identity(dist), lambda ddist: _grad(ddist, diff, dist)

@pi_named('cutoff_func')
def cutoff_func(dist, cutoff_type='f1', rc=5.0):
	"""returns the cutoff function of given type

	Args:
		dist (tensor): a tensor of distance
		cutoff_type (string): name of the cutoff function
		rc (float): cutoff radius

	Returns: 
		A cutoff function tensor with the same shape of dist
	"""
	cutoff_fn = {'f1': lambda x: 0.5*(tf.math.cos(np.pi*x/rc)+1),
				 'f2': lambda x: (tf.math.tanh(1-x/rc)/np.tanh(1))**3,
				 'hip': lambda x: tf.math.cos(np.pi*x/rc/2)**2}
	return cutoff_fn[cutoff_type](dist)




if __name__ == '__main__':
	pass
