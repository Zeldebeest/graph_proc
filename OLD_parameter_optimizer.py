""" Tools for optimizing the parameters of the images in a graph to minimize the overall residual.
**Author:**   Oliver Zeldin <zeldin@stanford.edu>
"""
from __future__ import division
__author__ = 'zeldin'
import logging
from scipy.optimize import fmin_l_bfgs_b as lbfgs

def loss_func(x):
  """ return the contribution to the loss function of residual x. Placeholder for rebust methods to come """
  return x**2

def total_score(params, *args):
  """ This is the function to be minimized by graph processing.

  It returns the total residual given params, which is a tuple containing the parameters to refine for one image. Formated for use with the scipy minimization routines. There are thus between 5 and 12,

  :param params: a list of the parameters for the minimisation as a tuple, only including vertices that have edges. See ImageNode.get_x0() for details of parameters.
  :param *args:  a 2-tuple of the vertex to be minimised, cross_val. cross_val is a list of edge to leave out of the optimisation for cross-validation purposes.

  :return: the total squared residual of the graph minimisation
  """

  vertex = args[0]
  cross_val = args[1]

  # 1. Update scales and partialities for each node that has edges
  # using params.
  vertex.partialties = vertex.calc_partiality(params)
  vertex.scales = vertex.calc_scales(params)
  vertex.G = params[0]
  vertex.B = params[1]

  # 2. Calculate all new residuals
  residuals_by_edge = []
  for edge in vertex.edges:
    if edge not in cross_val:
      residuals_by_edge.append(edge.residuals())

  # 3. Return the sum squared of all residuals
  total_sum = 0
  for edge in residuals_by_edge:
    for residual in edge:
      total_sum += loss_func(residual)

  logging.debug("Total Score: {}".format(total_sum))
  return total_sum


def _calc_residuals(work_edges, test_edges):
  """ Internal function to quickly calculate the sum of residuals for two lists of edges: work edges, and test edges. """

  work_residual = 0
  test_residual = 0
  # Calculate total graph residual
  for e in work_edges:
      for r in e.residuals():
        work_residual += loss_func(r)

  for e in test_edges:
      for r in e.residuals():
        test_residual += loss_func(r)

  return work_residual, test_residual

def global_minimise(graph, nsteps=10, eta=10, cross_val=[None]):
  """
  Perform a global minimisation on the total squared residuals on all the edges, as calculated by Edge.residuals(). Uses the L-BFGS algorithm.

  Done by repetedly locally optimizing each node, and repeating this until convergence.

  :param graph: the graph object to minimize.
  :param nsteps: max number of iterations.
  :param cross_val: a list of edge to leave out of the optimisation for cross-validation purposes.

  :return: (end_residual, starting_residual) for the overall graph
  """

  # make test/work set
  work_edges = []
  test_edges = []
  for e in graph.edges:
    if e in cross_val:
      test_edges.append(e)
    else:
      work_edges.append(e)

  init_work_residual, init_test_residual = _calc_residuals(work_edges,
                                                           test_edges)
  logging.info("Starting work/test residual is {}/{}".format(init_work_residual,
                                                            init_test_residual))

  old_residual = float("inf")
  new_residual = init_work_residual
  n_int = 0
  work_residuals = [init_work_residual]
  test_residuals = [init_test_residual]
  param_history = [[v.params for v in graph.members]]
  while abs(old_residual - new_residual) > eta and n_int < nsteps:
    new_params = []
    for v in graph.members:
      final_params, min_total_res, info_dict = lbfgs(total_score,
                                                    list(v.get_x0()),
                                                     approx_grad=True,
                                                     epsilon=0.001,
                                                     args=[v, cross_val],
                                                     factr=10**12,
                                                     iprint=0,
                                                     disp=10)
      new_params.append((v, final_params))




    # Update scales and partialities for each node that has edges.
    for v, final_params in new_params:
      v.params = final_params
      v.partialties = v.calc_partiality(final_params)
      v.scales = v.calc_scales(final_params)
      v.G = final_params[0]
      v.B = final_params[1]
      if any(v.partialities == v.calc_partiality(v.get_x0())):
        pchange = False
      else:
        pchange = True
      if any(v.scales == v.calc_scales(v.get_x0())):
        schange = False
      else:
        schange = True
      logging.info("partialityi/scales changed: {}/{}".format(pchange,schange))

    param_history.append([v.params for v in graph.members])
    work_residual, test_residual = _calc_residuals(work_edges,
                                                   test_edges)
    work_residuals.append(work_residual)
    test_residuals.append(test_residual)
    n_int += 1
    logging.info("Iteration {}: work/test residual is {}/{}".format(n_int,
                                                                  work_residual,
                                                                  test_residual))
    if n_int == nsteps:
      logging.warning("Reached max iterations")

    old_residual = new_residual
    new_residual = work_residual

  return work_residuals, test_residuals, param_history
