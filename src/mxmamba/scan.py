from math import log2, ceil
import mlx.core as mx


@mx.compile
def scan(deltaA: mx.array, deltaB_u: mx.array, C: mx.array) -> mx.array:
    """
    Straightforward serial implementation of Mamba's selective scan.

    Args:
        deltaA: shape (b, l, d, n)
        deltaB_u: shape (b, l, d, n)
        C: shape (b, l, n)

    Returns:
        shape (b, l, d)
    """
    b, l, d, n = deltaA.shape
    x = mx.zeros((b, d, n))
    y = mx.zeros((b, l, d))

    for i in range(l):
        x = deltaA[:, i] * x + deltaB_u[:, i]
        # einsum('bdn, bn -> bd', x, C[:, i])
        y[:, i] = (x * mx.expand_dims(C[:, i], 1)).sum(2)

    return y


# @mx.compile
def parallel_scan(deltaA: mx.array, deltaB_u: mx.array, C: mx.array) -> mx.array:
    """
    Parallel implementation of scan as described in [1].

    [1]: Blelloch, Guy E. "Prefix sums and their applications." (1990)
    """
    b, l, d, n = deltaA.shape
    if (l & (l - 1)) != 0:
        l = 2 ** ceil(log2(l))

    # Intermediate state containing the products of A bar
    xA = mx.zeros((b, l, d, n), dtype=deltaA.dtype)
    # xA[:, 0] = 1.
    # xA[:, 1:deltaA.shape[1] + 1] = deltaA
    xA[:, :deltaA.shape[1]] = deltaA
    # Latent (scan) state
    x = mx.zeros((b, l, d, n), dtype=deltaA.dtype)
    x[:, :deltaA.shape[1]] = deltaB_u

    lastA, lastB_u = xA[:, -1], x[:, -1]
    levels = ceil(log2(l))

    # up-sweep
    for d in range(levels):
        stride = 2 ** (d + 1)
        for k in range(0, l, stride):
            left, right = k + 2**d - 1, k + stride - 1
            x[:, right] += xA[:, right] * x[:, left]
            xA[:, right] *= xA[:, left]

    # down-sweep
    x[:, -1] = 0.
    xA[:, -1] = 1.
    for d in reversed(range(levels)):
        stride = 2 ** (d + 1)
        for k in range(0, l, stride):
            left, right = k + 2**d - 1, k + stride - 1
            x[:, left], x[:, right] = (x[:, right],
                                       xA[:, left] * x[:, right] + x[:, left])
            xA[:, left], xA[:, right] = xA[:, right], xA[:, left] * xA[:, right]

    # convert exclusive scan to inclusive scan
    x[:, :l-1] = x[:, 1:]
    x[:, -1] = (x[:, -1] * lastA) + lastB_u
    x = x[:, :deltaA.shape[1]]

    # x (b, l, d, n)
    # C (b, l, n)
    y = (x * mx.expand_dims(C, 2)).sum(3)

    return y

# Reference 1D scan implementations


def scan1d(arr):
    res = mx.zeros_like(arr)
    acc = 0
    for i in range(len(res)):
        acc += arr[i]
        res[i] = acc
    return res


@mx.compile
def parallel_scan1d_upsweep(arr):
    n = arr.shape[0]
    for d in range(ceil(log2(n))):
        stride = 2 ** (d + 1)
        for k in range(0, n, stride):
            arr[k + stride - 1] += arr[k + 2**d - 1]
    return arr


@mx.compile
def parallel_scan1d_downsweep(arr):
    n = arr.shape[0]
    arr[-1] = 0
    for d in reversed(range(ceil(log2(n)))):
        stride = 2 ** (d + 1)
        for k in range(0, n, stride):
            left, right = k + 2**d - 1, k + stride - 1
            arr[left], arr[right] = arr[right], arr[left] + arr[right]
    return arr


def parallel_scan1d(arr):
    arr_n = n = arr.shape[0]
    res = mx.array(arr)

    if (n & (n - 1)) == 0:
        res = mx.array(arr)
    else:
        # Pad array length to next power of 2
        n = (2 ** ceil(log2(n)))
        res = mx.zeros(n, dtype=arr.dtype)
        res[:arr_n] = arr

    last = res[-1]

    res = parallel_scan1d_upsweep(res)
    res = parallel_scan1d_downsweep(res)

    # convert exclusive scan to inclusive scan
    res[:n-1] = res[1:]
    res[-1] += last

    return res[:arr_n]
