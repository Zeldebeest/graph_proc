""" Tools for optimizing the parameters of the images in a graph to minimize the overall residual.
**Author:**   Oliver Zeldin <zeldin@stanford.edu>
"""
from __future__ import division
__author__ = 'zeldin'
import logging
import numpy as np

def total_score(params, *args):
  """ This is the function to be minimized by graph processing.

  It returns the total residual given params, which is a list of tuples, each
  containing the parameters to refine for one image. Formated for use with
  the scipy minimization routines.
  There are thus 12*N_images parameters, and sum_edges(edge_weight) variables

  :param params: a list of the parameters for the minimisation as a tuple, only including vertices that have edges. See ImageNode.get_x0() for details of parameters.
  :param *args: is the Graph to be minimised, the starting point of params for each new frame, as a list, and a list of the vertex objects that are to be used.
  :return: the total squared residual of the graph minimisation
  """
  assert len(args) == 3, "Must have 2 args: Graph object, and length of each" \
                         "set of parameters."

  graph = args[0]
  param_knots = args[1]
  x0 = args[2]

  # Scale the normalised params up by their starting values.
  params = params * x0

  # 0. break params up into a list of correct length
  param_list = []
  current_pos = 0
  for x0_length in param_knots:
    param_list.append(params[current_pos:current_pos + x0_length])
    current_pos += x0_length
  params = param_list

  # 1. Update scales and partialities for each node that has edges
  # using params.
  for v_id, vertex in enumerate([vertex for vertex in graph.members
                                 if vertex.edges]):
    vertex.partialties = vertex.calc_partiality(params[v_id])
    vertex.scales = vertex.calc_scales(params[v_id])
    vertex.G = params[v_id][0]
    vertex.B = params[v_id][1]

  # 2. Calculate all new residuals
  residuals_by_edge = []
  for edge in graph.edges:
    residuals_by_edge.append(edge.residuals())

  # 3. Return the sum squared of all residuals
  total_sum = 0
  for edge in residuals_by_edge:
    for residual in edge:
      total_sum += residual**2

  logging.debug("Total Score: {}".format(total_sum))
  return total_sum

def global_minimise(graph, nsteps=15000):
  """
  Perform a global minimisation on the total squared residuals on all the
  edges, as calculated by Edge.residuals(). Uses the L-BFGS algorithm.

  :param graph: the graph object to minimize.
  :param nsteps: max number of iterations for L-BFGS to use.
  """
  from scipy.optimize import fmin_l_bfgs_b as lbfgs

  # 1. Make a big array of all x0 values
  x0 = []
  param_knots = []
  for vertex in graph.members:
    if vertex.edges:
      x0.extend(vertex.get_x0())
      param_knots.append(len(vertex.get_x0()))

  # Test for the above:
  current_pos = 0
  vertices_with_edges = [vert for vert in graph.members if vert.edges]
  for im_num, x0_length in enumerate(param_knots):
    these_params = x0[current_pos:current_pos+ x0_length]
    assert all(these_params == vertices_with_edges[im_num].get_x0())
    current_pos += x0_length

  # 2. Do the magic
  final_params, min_total_res, info_dict = lbfgs(total_score,
                                                 x0,
                                                 approx_grad=True,
                                                 epsilon=0.001,
                                                 args=(graph, param_knots,
                                                       np.ones(len(x0))),
                                                 factr=10**12,
                                                 iprint=0,
                                                 disp=10,
                                                 maxiter=nsteps)

  return final_params, min_total_res, info_dict
