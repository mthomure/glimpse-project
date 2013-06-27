# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

# Adapted from sklearn.cluster.kmeans

import warnings

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import _k_means
from sklearn.cluster.k_means_ import _squared_norms, _k_init, MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.sparsefuncs import mean_variance_axis0
from sklearn.utils import check_arrays
from sklearn.utils import check_random_state
from sklearn.utils import atleast2d_or_csr
from sklearn.utils import as_float_array
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

###############################################################################

def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset"""

    # [MICK] should the variance be replaced with the weighted sum of variation from the mean?

    variances = np.var(X, axis=0)  # average variance per-feature
    return np.mean(variances) * tol  # result is percent of mean feature variance

def _labels_inertia_precompute_dense(X, x_squared_norms, centers, weights=None):
    n_samples = X.shape[0]
    k = centers.shape[0]
    # TODO: if X is sphered, use dot product instead.
    distances = euclidean_distances(centers, X, x_squared_norms,
                                    squared=True)
    labels = np.empty(n_samples, dtype=np.int32)
    labels.fill(-1)
    mindist = np.empty(n_samples)
    mindist.fill(np.infty)
    for center_id in range(k):
        dist = distances[center_id]
        labels[dist < mindist] = center_id
        mindist = np.minimum(dist, mindist)

    # [MICK] updated to use weights
    if weights is None:
      inertia = mindist.sum()
    else:
      inertia = (mindist * weights[:,np.newaxis]).sum()

    return labels, inertia

def _labels_inertia(X, x_squared_norms, centers, precompute_distances=True,
                    distances=None, weights=None):
    """E step of the K-means EM algorithm

    Compute the labels and the inertia of the given samples and centers

    Parameters
    ----------
    X: float64 array-like or CSR sparse matrix, shape (n_samples, n_features)
        The input samples to assign to the labels.

    x_squared_norms: array, shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers: float64 array, shape (k, n_features)
        The cluster centers.

    distances: float64 array, shape (n_samples,)
        Distances for each sample to its closest center.

    Returns
    -------
    labels: int array of shape(n)
        The resulting assignment

    inertia: float
        The value of the inertia criterion with the assignment
    """
    n_samples = X.shape[0]
    # set the default value of centers to -1 to be able to detect any anomaly
    # easily
    labels = - np.ones(n_samples, np.int32)
    if distances is None:
        distances = np.zeros(shape=(0,), dtype=np.float64)
    assert precompute_distances, "Disabling precompute_distances is unsupported"
    return _labels_inertia_precompute_dense(X, x_squared_norms, centers,
            weights)

def _init_centroids(X, k, init, random_state=None, x_squared_norms=None,
                    init_size=None, weights=None, sphered=False):
    """Compute the initial centroids

    Parameters
    ----------

    X: array, shape (n_samples, n_features)

    k: int
        number of centroids

    init: {'k-means++', 'random' or ndarray or callable} optional
        Method for initialization

    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    x_squared_norms:  array, shape (n_samples,), optional
        Squared euclidean norm of each data point. Pass it if you have it at
        hands already to avoid it being recomputed here. Default: None

    init_size : int, optional
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accurracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than k.

    Returns
    -------
    centers: array, shape(k, n_features)
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if init_size is not None and init_size < n_samples:
        if init_size < k:
            warnings.warn(
                "init_size=%d should be larger than k=%d. "
                "Setting it to 3*k" % (init_size, k),
                RuntimeWarning, stacklevel=2)
            init_size = 3 * k
        init_indices = random_state.random_integers(
            0, n_samples - 1, init_size)
        X = X[init_indices]
        weights = weights[init_indices] if weights is not None else None
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]
    elif n_samples < k:
            raise ValueError(
                "n_samples=%d should be larger than k=%d" % (n_samples, k))

    if init == 'k-means++':
        assert weights is None and not sphered, "k-means++ initialization is not supported for weighted or sphered data."
        centers = _k_init(X, k, random_state=random_state,
                          x_squared_norms=x_squared_norms)
    elif init == 'random':
        seeds = random_state.permutation(n_samples)[:k]
        centers = X[seeds]
    elif hasattr(init, '__array__'):
        centers = init
    elif callable(init):
        centers = init(X, k, random_state=random_state, weights=weights)
    else:
        raise ValueError("the init parameter for the k-means should "
                         "be 'k-means++' or 'random' or an ndarray, "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    if sp.issparse(centers):
        centers = centers.toarray()

    if len(centers) != k:
        raise ValueError('The shape of the inital centers (%s) '
                         'does not match the number of clusters %i'
                         % (centers.shape, k))

    return centers

def _mini_batch_convergence(model, iteration_idx, n_iter, tol,
                            n_samples, centers_squared_diff, batch_inertia,
                            context, verbose=0):
    """Helper function to encapsulte the early stopping logic"""
    # Normalize inertia to be able to compare values when
    # batch_size changes
    batch_inertia /= model.batch_size
    centers_squared_diff /= model.batch_size

    # Compute an Exponentially Weighted Average of the squared
    # diff to monitor the convergence while discarding
    # minibatch-local stochastic variability:
    # https://en.wikipedia.org/wiki/Moving_average
    ewa_diff = context.get('ewa_diff')
    ewa_inertia = context.get('ewa_inertia')
    if ewa_diff is None:
        ewa_diff = centers_squared_diff
        ewa_inertia = batch_inertia
    else:
        alpha = float(model.batch_size) * 2.0 / (n_samples + 1)
        alpha = 1.0 if alpha > 1.0 else alpha
        ewa_diff = ewa_diff * (1 - alpha) + centers_squared_diff * alpha
        ewa_inertia = ewa_inertia * (1 - alpha) + batch_inertia * alpha

    # Log progress to be able to monitor convergence
    if verbose:
        progress_msg = (
            'Minibatch iteration %d/%d:'
            'mean batch inertia: %f, ewa inertia: %f ' % (
                iteration_idx + 1, n_iter, batch_inertia,
                ewa_inertia))
        print progress_msg

    # Early stopping based on absolute tolerance on squared change of
    # centers postion (using EWA smoothing)
    if tol > 0.0 and ewa_diff < tol:
        if verbose:
            print 'Converged (small centers change) at iteration %d/%d' % (
                iteration_idx + 1, n_iter)
        return True

    # Early stopping heuristic due to lack of improvement on smoothed inertia
    ewa_inertia_min = context.get('ewa_inertia_min')
    no_improvement = context.get('no_improvement', 0)
    if (ewa_inertia_min is None or ewa_inertia < ewa_inertia_min):
        no_improvement = 0
        ewa_inertia_min = ewa_inertia
    else:
        no_improvement += 1

    if (model.max_no_improvement is not None
            and no_improvement >= model.max_no_improvement):
        if verbose:
            print ('Converged (lack of improvement in inertia)'
                   ' at iteration %d/%d' % (
                       iteration_idx + 1, n_iter))
        return True

    # update the convergence context to maintain state across sucessive calls:
    context['ewa_diff'] = ewa_diff
    context['ewa_inertia'] = ewa_inertia
    context['ewa_inertia_min'] = ewa_inertia_min
    context['no_improvement'] = no_improvement
    return False

def _mini_batch_step(X, x_squared_norms, centers, counts,
                     old_center_buffer, compute_squared_diff,
                     distances=None, random_reassign=False,
                     random_state=None, reassignment_ratio=.01,
                     verbose=False, weights=None, sphered=False, epsilon_norm=0):
    """Incremental update of the centers for the Minibatch K-Means algorithm

    Parameters
    ----------

    X: array, shape (n_samples, n_features)
        The original data array.

    x_squared_norms: array, shape (n_samples,)
        Squared euclidean norm of each data point.

    centers: array, shape (k, n_features)
        The cluster centers. This array is MODIFIED IN PLACE

    counts: array, shape (k,)
         The vector in which we keep track of the numbers of elements in a
         cluster. This array is MODIFIED IN PLACE

    distances: array, dtype float64, shape (n_samples), optional
        If not None, should be a pre-allocated array that will be used to store
        the distances of each sample to it's closest center.

    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    random_reassign: boolean, optional
        If True, centers with very low counts are
        randomly-reassigned to observations in dense areas.

    reassignment_ratio: float, optional
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more easily reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    verbose: bool, optional
        Controls the verbosity

    """
    # Perform label assignement to nearest centers
    nearest_center, inertia = _labels_inertia(X, x_squared_norms, centers,
                                              distances=distances, weights=weights)
    if random_reassign and reassignment_ratio > 0:
        random_state = check_random_state(random_state)
        # Reassign clusters that have very low counts
        to_reassign = np.logical_or(
            (counts <= 1), counts <= reassignment_ratio * counts.max())
        number_of_reassignments = to_reassign.sum()
        if number_of_reassignments:

            # TODO: extend this to deal with weighted and sphered data.
            assert weights is None, "Random reassignment is unsupported for weighted data."

            # Pick new clusters amongst observations with a probability
            # proportional to their closeness to their center
            distance_to_centers = np.asarray(centers[nearest_center] - X)
            distance_to_centers **= 2
            distance_to_centers = distance_to_centers.sum(axis=1)
            # Flip the ordering of the distances
            distance_to_centers -= distance_to_centers.max()
            distance_to_centers *= -1
            rand_vals = random_state.rand(number_of_reassignments)
            rand_vals *= distance_to_centers.sum()
            new_centers = np.searchsorted(distance_to_centers.cumsum(),
                                          rand_vals)
            new_centers = X[new_centers]
            if verbose:
                n_reassigns = to_reassign.sum()
                if n_reassigns:
                    print("[_mini_batch_step] Reassigning %i cluster centers."
                          % n_reassigns)
            if sp.issparse(new_centers) and not sp.issparse(centers):
                new_centers = new_centers.toarray()

            if sphered:
                for c in new_centers: c /= np.linalg.norm(c) + epsilon_norm

            centers[to_reassign] = new_centers

    # implementation for the sparse CSR reprensation completely written in
    # cython
    if sp.issparse(X):
        return inertia, _k_means._mini_batch_update_csr(
            X, x_squared_norms, centers, counts, nearest_center,
            old_center_buffer, compute_squared_diff)

    # dense variant in mostly numpy (not as memory efficient though)
    k = centers.shape[0]
    squared_diff = 0.0
    for center_idx in range(k):
        # find points from minibatch that are assigned to this center
        center_mask = nearest_center == center_idx
        count = center_mask.sum()

        if count <= 0:
            continue

        if weights is not None:
            count = weights[center_mask].sum()  # update "count" to be sum of weights
            if count <= 0:
                continue;

        center = centers[center_idx]

        if compute_squared_diff:
            old_center_buffer[:] = center

        # inplace remove previous count scaling
        centers[center_idx] *= counts[center_idx]

        # inplace sum with new points members of this cluster
        # [MICK] updated to include weights
        if weights is None:
            center += np.sum(X[center_mask], axis=0)
        else:
            center += np.sum(X[center_mask] * weights[center_mask][:,np.newaxis], axis=0)

        if sphered:
          center /= np.linalg.norm(center)

        # update the count statistics for this center
        counts[center_idx] += count

        # inplace rescale to compute mean of all points (old and new)
        center /= counts[center_idx]

        # update the squared diff if necessary
        if compute_squared_diff:
            squared_diff += np.sum(
                (center - old_center_buffer) ** 2)

    return inertia, squared_diff


class WeightedMiniBatchKMeans(MiniBatchKMeans):
    """Mini-Batch K-Means clustering with weighted instances."""

    def __init__(self, n_clusters=8, init='k-means++', max_iter=100,
                 batch_size=100, verbose=0, compute_labels=True,
                 random_state=None, tol=0.0, max_no_improvement=10,
                 init_size=None, n_init=3, k=None,
                 reassignment_ratio=0.01, sphered=False):

        super(MiniBatchKMeans, self).__init__(
            n_clusters=n_clusters, init=init, max_iter=max_iter,
            verbose=verbose, random_state=random_state, tol=tol, n_init=n_init,
            k=k)

        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.compute_labels = compute_labels
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio

        self.initial_cluster_centers_ = None
        self.sphered = sphered

    def fit(self, X, y=None, weights=None):
        """Compute the centroids on X by chunking it into mini-batches.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Coordinates of the data points to cluster
        """
        random_state = check_random_state(self.random_state)
        if self.k is not None:
            warnings.warn("Parameter k has been replaced by 'n_clusters'"
                          " and will be removed in release 0.14.",
                          DeprecationWarning, stacklevel=2)
            self.n_clusters = self.k
        X = check_arrays(X, sparse_format="csr", copy=False,
                         check_ccontiguous=True, dtype=np.float64)[0]
        if sp.issparse(X):
            raise ValueError("Only dense arrays are supported")

        n_samples, n_features = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("Number of samples smaller than number "
                             "of clusters.")

        if hasattr(self.init, '__array__'):
            self.init = np.ascontiguousarray(self.init, dtype=np.float64)

        x_squared_norms = _squared_norms(X)

        if self.tol > 0.0:
            tol = _tolerance(X, self.tol)

            # using tol-based early stopping needs the allocation of a
            # dedicated before which can be expensive for high dim data:
            # hence we allocate it outside of the main loop
            old_center_buffer = np.zeros(n_features, np.double)
        else:
            tol = 0.0
            # no need for the center buffer if tol-based early stopping is
            # disabled
            old_center_buffer = np.zeros(0, np.double)

        distances = np.zeros(self.batch_size, dtype=np.float64)
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        n_iter = int(self.max_iter * n_batches)

        init_size = self.init_size
        if init_size is None:
            init_size = 3 * self.batch_size
        if init_size > n_samples:
            init_size = n_samples
        self.init_size_ = init_size

        validation_indices = random_state.random_integers(
            0, n_samples - 1, init_size)
        X_valid = X[validation_indices]
        x_squared_norms_valid = x_squared_norms[validation_indices]
        weights_valid = weights[validation_indices] if weights is not None else None

        # perform several inits with random sub-sets
        best_inertia = None
        for init_idx in range(self.n_init):
            if self.verbose:
                print "Init %d/%d with method: %s" % (
                    init_idx + 1, self.n_init, self.init)
            counts = np.zeros(self.n_clusters, dtype=np.int32)

            # TODO: once the `k_means` function works with sparse input we
            # should refactor the following init to use it instead.

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans
            cluster_centers = _init_centroids(
                X, self.n_clusters, self.init,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                init_size=init_size, weights=weights, sphered=self.sphered)

            # Compute the label assignement on the init dataset
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X_valid, x_squared_norms[validation_indices],
                cluster_centers, counts, old_center_buffer, False,
                distances=distances, verbose=self.verbose,
                weights=weights_valid, sphered=self.sphered)

            # Keep only the best cluster centers across independent inits on
            # the common validation set
            _, inertia = _labels_inertia(X_valid, x_squared_norms_valid,
                                         cluster_centers, weights=weights)
            if self.verbose:
                print "Inertia for init %d/%d: %f" % (
                    init_idx + 1, self.n_init, inertia)
            if best_inertia is None or inertia < best_inertia:
                self.cluster_centers_ = cluster_centers
                self.counts_ = counts
                best_inertia = inertia

        # Empty context to be used inplace by the convergence check routine
        convergence_context = {}

        self.initial_cluster_centers_ = self.cluster_centers_.copy()

        # Perform the iterative optimization until the final convergence
        # criterion
        for iteration_idx in xrange(n_iter):

            # Sample a minibatch from the full dataset
            minibatch_indices = random_state.random_integers(
                0, n_samples - 1, self.batch_size)

            minibatch_weights = weights[minibatch_indices] if weights is not None else None

            # Perform the actual update step on the minibatch data
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X[minibatch_indices], x_squared_norms[minibatch_indices],
                self.cluster_centers_, self.counts_,
                old_center_buffer, tol > 0.0, distances=distances,
                # Here we randomly choose whether to perform
                # random reassignment: the choice is done as a function
                # of the iteration index, and the minimum number of
                # counts, in order to force this reassignment to happen
                # every once in a while
                random_reassign=((iteration_idx + 1)
                                 % (10 + self.counts_.min()) == 0),
                random_state=random_state,
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose, weights=minibatch_weights, sphered=self.sphered)

            # Monitor convergence and do early stopping if necessary
            if _mini_batch_convergence(
                    self, iteration_idx, n_iter, tol, n_samples,
                    centers_squared_diff, batch_inertia, convergence_context,
                    verbose=self.verbose):
                break

        if self.compute_labels:
            if self.verbose:
                print 'Computing label assignements and total inertia'
            self.labels_, self.inertia_ = _labels_inertia(
                X, x_squared_norms, self.cluster_centers_, weights=weights)

        return self

    def partial_fit(self, X, y=None, weights=None):
        """Update k means estimate on a single mini-batch X.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Coordinates of the data points to cluster.
        """

        X = check_arrays(X, sparse_format="csr", copy=False)[0]
        n_samples, n_features = X.shape
        if hasattr(self.init, '__array__'):
            self.init = np.ascontiguousarray(self.init, dtype=np.float64)

        if n_samples == 0:
            return self

        x_squared_norms = _squared_norms(X)
        self.random_state_ = check_random_state(self.random_state)
        if (not hasattr(self, 'counts_')
                or not hasattr(self, 'cluster_centers_')):
            # this is the first call partial_fit on this object:
            # initialize the cluster centers
            self.cluster_centers_ = _init_centroids(
                X, self.n_clusters, self.init,
                random_state=self.random_state_,
                x_squared_norms=x_squared_norms, init_size=self.init_size, weights=weights)

            self.initial_cluster_centers_ = self.cluster_centers_.copy()

            self.counts_ = np.zeros(self.n_clusters, dtype=np.int32)
            random_reassign = False
        else:
            # The lower the minimum count is, the more we do random
            # reassignment, however, we don't want to do random
            # reassignment too often, to allow for building up counts
            random_reassign = self.random_state_.randint(
                10 * (1 + self.counts_.min())) == 0

        _mini_batch_step(X, x_squared_norms, self.cluster_centers_,
                         self.counts_, np.zeros(0, np.double), 0,
                         random_reassign=random_reassign,
                         random_state=self.random_state_,
                         reassignment_ratio=self.reassignment_ratio,
                         verbose=self.verbose, weights=weights, sphered=self.sphered)

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia(
                X, x_squared_norms, self.cluster_centers_, weights=weights)

        return self
