import torch
import torch.nn.functional as F
import warnings
import numpy as np


def th_cov(x):
    """
    :param x: N x M x D
    :return: N x D x D
    """

    x = x - torch.mean(x, dim=1, keepdim=True) # (N, M, D)
    cov = torch.bmm(x.permute(0, 2, 1), x) / x.size(1) # (N, D, D)
    return cov


def cdist(x, y):
    """
    :param x: (N, M, D)
    :param y: (N, k, D)
    :return:  (N, M, K)
    """
    if x.dim() == 2:
        x = x.unsqueeze(dim=2)
        y = y.unsqueeze(dim=2)
    dist0 = torch.sum(x ** 2.0, dim=-1)      # (N, M)
    dist1 = torch.sum(y ** 2.0, dim=-1)      # (N, k)
    dist2 = torch.bmm(x, y.permute(0, 2, 1)) # (N, M, k)
    dist  = dist0.unsqueeze(dim=2) + dist1.unsqueeze(dim=1) - 2.0 * dist2 # (N, M, k)
    return dist


class ClusterError(Exception):
    pass


def _missing_warn():
    """Print a warning when called."""
    warnings.warn("One of the clusters is empty. "
                  "Re-run kmeans with a different initialization.")


def _missing_raise():
    """raise a ClusterError when called."""
    raise ClusterError("One of the clusters is empty. "
                       "Re-run kmeans with a different initialization.")


_valid_miss_meth = {'warn': _missing_warn, 'raise': _missing_raise}


def _kpoints(data, k):
    """
    data : ndarray
        A N x M x D array
    k : int
        Number of samples to generate.
   Returns
    -------
    x : ndarray
        A N x k x D containing the initial centroids
    """
    idx  = torch.stack([torch.randperm(data.size(1), device=data.device)[:k]
                        for _ in range(data.size(0))], dim=0)   # (N, k)
    if data.dim() > 2:
        idx = idx.unsqueeze(dim=2).expand(-1, -1, data.size(2)) # (N, k, D)
    init = torch.gather(data, dim=1, index=idx)
    return init


# def _kpoints2(data, k):
#     """
#     data : ndarray
#         A N x M x D array
#     k : int
#         Number of samples to generate.
#    Returns
#     -------
#     x : ndarray
#         A N x k x D containing the initial centroids
#     """
#     idx  = torch.stack([torch.randperm(data.size(1))[:k] + i * data.size(1) for i in range(data.size(0))], dim=0)   # (N*k)
#     data = data.view(-1, data.size(2))
#     init = data[idx]
#     init = init.view(-1, k, data.size(1))
#     return init


def _krandinit(data, k):
    """Returns k samples of a random variable which parameters depend on data.

    More precisely, it returns k observations sampled from a Gaussian random
    variable which mean and covariances are the one estimated from data.

    Parameters
    ----------
    data : ndarray N x M x D
    k : int
        Number of samples to generate.

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids
    """
    mu = torch.mean(data, dim=1, keepdim=True)                              # (N, 1, D)

    if data.dim() == 2:
        std = torch.std(data, dim=1, keepdim=True)                          # (N, 1)
        x   = torch.randn((data.size(0), k), dtype=data.dtype, device=data.device) # (N, k)
        x  *= std                                                           # (N, k)
    elif data.size(2) > data.size(1):
        # initialize when the covariance matrix is rank deficient
        _, s, v = torch.svd(data - mu, some=True)                           # (N, M, M)  (N, M)  (N, D, M)
        x   = torch.randn((s.size(0), k, s.size(1)), dtype=s.dtype, device=s.device) # (N, k, M)
        sVh = s[..., None] * v.permute(0, 2, 1) / np.sqrt(data.size(1) - 1) # (N, M, D) = (N, M, 1) * (N, M, D) / (M - 1)
        x   = torch.bmm(x, sVh)                                             # (N, k, D) = (N, k, M) x (N, M, D)
    else:
        cov = th_cov(data)                                                  # (N, D, D)
        eye = torch.eye(data.size(2), dtype=data.dtype, device=data.device) # (D, D)
        cov = cov + eye * 1e-8
        x   = torch.randn((data.size(0), k, data.size(2)), dtype=data.dtype, device=data.device) # (N, k, D)
        x   = torch.bmm(x, torch.cholesky(cov, upper=True))                 # (N, k, D) = (N, k, D) x (N, D, D)
    x += mu
    return x


def _kpp(data, k):
    """ Picks k points in data based on the kmeans++ method

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
        dimensional data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.

    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    """
    if data.dim() > 2:
        init = torch.zeros((data.size(0), k, data.size(2)), dtype=data.dtype, device=data.device)
    else:
        init = torch.zeros((data.size(0), k), dtype=data.dtype, device=data.device)

    for i in range(k):
        if i == 0:
            idx = torch.randint(data.size(1), (data.size(0), 1), device=data.device)  # (N, 1)
        else:
            D2  = torch.min(cdist(data, init[:, :i]), dim=2)[0]      # (N, M)
            probs = D2 / (D2.sum(dim=1, keepdim=True) + 1e-8)        # (N, M)
            cumprobs = torch.cumsum(probs, dim=1)                    # (N, M)
            r   = torch.rand((data.size(0), 1), dtype=data.dtype, device=data.device) # (N, 1)
            cumprobs = torch.cat([cumprobs, r], dim=1)               # (N, M+1)
            idx = torch.argsort(cumprobs, dim=1)                     # (N, M+1)
            idx = torch.nonzero(idx == cumprobs.size(1)-1)[:, 1:2]   # (N, 1)
            idx[idx == cumprobs.size(1)-1] = cumprobs.size(1)-2      # (N, 1)
        if data.dim() > 2:
            idx = idx.unsqueeze(dim=2).expand(-1, -1, data.size(2))  # (N, 1, D)
        init[:, i:i+1] = torch.gather(data, dim=1, index=idx)        # (N, k, D)
    return init


_valid_init_meth = {'random': _krandinit, 'points': _kpoints, '++': _kpp}


def whiten(obs, check_finite=True):
    """
    Normalize a group of observations on a per feature basis.

    Before running k-means, it is beneficial to rescale each feature
    dimension of the observation set with whitening. Each feature is
    divided by its standard deviation across all observations to give
    it unit variance.

    Parameters
    ----------
    obs : ndarray N x M x D
        Each row of the array is an observation.  The
        columns are the features seen during each observation.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    Returns
    -------
    result : ndarray
        Contains the values in `obs` scaled by the standard deviation of each column.
    """
    std_dev = torch.std(obs, dim=1, keepdim=True) # (N, 1, D)
    zero_std_mask = std_dev == 0
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
        warnings.warn("Some columns have standard deviation zero. "
                      "The values of these columns will not change.",
                      RuntimeWarning)
    return obs / std_dev


# def vq(obs, code_book, check_finite=True):
#     """
#     The algorithm computes the euclidian distance between each
#     observation and every frame in the code_book.
#     Returns
#     -------
#     code : ndarray
#         code[i] gives the label of the ith obversation, that its code is code_book[code[i]].
#     mind_dist : ndarray
#         min_dist[i] gives the distance between the ith observation and its corresponding code.
#     """
#     if obs.dim() != code_book.dim():
#         raise ValueError("Observation and code_book should have the same rank")
#
#     if obs.dim() == 2:
#         obs = obs.unsqueeze(dim=-1)
#         code_book = code_book.unsqueeze(dim=-1)
#
#     dist = cdist(obs, code_book)
#     min_dist, code = torch.min(dist, dim=-1)
#     return code, min_dist


def vq(obs, code_book, check_finite=True):
    """
    The algorithm computes the euclidian distance between each
    observation and every frame in the code_book.
    Returns
    -------
    code : ndarray
        code[i] gives the label of the ith obversation, that its code is code_book[code[i]].
    mind_dist : ndarray
        min_dist[i] gives the distance between the ith observation and its corresponding code.
    """
    if obs.dim() != code_book.dim():
        raise ValueError("Observation and code_book should have the same rank")

    if obs.dim() == 2:
        obs = obs.unsqueeze(dim=-1)
        code_book = code_book.unsqueeze(dim=-1)

    dist = cdist(obs, code_book)
    code = torch.argmin(dist, dim=-1)
    return code


def update_cluster_means(obs, labels, nc):
    """
    The update-step of K-means. Calculate the mean of observations in each
    cluster.
    Parameters
    ----------
    obs : ndarray
        N x M x D
    labels : ndarray
        N x M The label of each observation.
    nc : int
        The number of centroids.
    Returns
    -------
    cb : ndarray
        The new code book.
    has_members : ndarray
        A boolean array indicating which clusters have members.
    Notes
    -----
    The empty clusters will be set to all zeros and the curresponding elements
    in `has_members` will be `False`. The upper level function should decide
    how to deal with them.
    """
    if labels.dim() != 2:
        raise ValueError('labels must be an 2d array')
    labels = F.one_hot(labels, num_classes=nc).type_as(obs) # (N, M, k)
    obs_count = torch.sum(labels, dim=1)                    # (N, k)
    cb = torch.bmm(labels.permute(0, 2, 1), obs)            # (N, k, D)
    cb = cb / (obs_count.unsqueeze(dim=-1) + 1e-8)          # (N, k, D)
    has_members = obs_count > 0                             # (N, k)
    return cb, has_members


def kmeans2(data, k, iter=10, thresh=1e-5, minit='random', missing='warn', check_finite=True):

    if int(iter) < 1:
        raise ValueError("Invalid iter (%s), must be a positive integer." % iter)
    try:
        miss_meth = _valid_miss_meth[missing]
    except KeyError:
        raise ValueError("Unknown missing method %r" % (missing,))

    if data.dim() == 2:
        d = 1
    elif data.dim() == 3:
        d = data.size(2)
    else:
        raise ValueError("Input of rank > 3 or rank < 2 is not supported.")

    if data.numel() < 1:
        raise ValueError("Empty input is not supported.")

    # If k is not a single value it should be compatible with data's shape
    if minit == 'matrix' or not np.isscalar(k):
        code_book = torch.as_tensor(k, dtype=data.dtype, device=data.device)
        if data.dim() != code_book.dim():
            raise ValueError("k array doesn't match data rank")
        nc = code_book.size(1)
        if data.dim() > 2 and code_book.size(2) != d:
            raise ValueError("k array doesn't match data dimension")
    else:
        nc = int(k)
        if nc < 1:
            raise ValueError("Cannot ask kmeans2 for %d clusters (k was %s)" % (nc, k))
        elif nc != k:
            warnings.warn("k was not an integer, was converted.")
        try:
            init_meth = _valid_init_meth[minit]
        except KeyError:
            raise ValueError("Unknown init method %r" % (minit,))
        else:
            code_book = init_meth(data, k)

    for i in range(iter):
        # Compute the nearest neighbor for each obs using the current code book
        # label = vq(data, code_book)[0]
        label = vq(data, code_book)
        # Update the code book by computing centroids
        new_code_book, has_members = update_cluster_means(data, label, nc)
        if not has_members.all():
            miss_meth()
            # Set the empty clusters to their previous positions
            new_code_book[~has_members] = code_book[~has_members]
        code_book = new_code_book

    return code_book, label
