from functools import partial

import mlx.core as mx
import pytest

from mxmamba.scan import scan1d, parallel_scan1d, scan, parallel_scan

BATCH_SIZE = 4
SEQ_LEN = 128

allclose = partial(mx.allclose, rtol=1e-5, atol=1e-4)


@pytest.mark.benchmark(warmup=False)
def test_scan1d(benchmark):
    mx.random.seed(0)
    x = mx.random.randint(0, 10000, shape=(5000,))
    benchmark(scan1d, x)


@pytest.mark.benchmark(warmup=False)
def test_scan1d_compiled(benchmark):
    mx.random.seed(0)
    x = mx.random.randint(0, 10000, shape=(5000,))
    benchmark(mx.compile(scan1d), x)


@pytest.mark.benchmark(warmup=False)
def test_parallel_scan1d_compiled(benchmark):
    mx.random.seed(0)
    x = mx.random.randint(0, 10000, shape=(5000,))
    res = benchmark(parallel_scan1d, x)
    assert mx.array_equal(res, scan1d(x))


@pytest.mark.benchmark(warmup=False)
def test_scan(benchmark):
    mx.random.seed(0)
    D, N = 768, 16

    deltaA = mx.random.normal((BATCH_SIZE, SEQ_LEN, D, N))
    deltaB_u = mx.random.normal((BATCH_SIZE, SEQ_LEN, D, N))
    C = mx.random.normal((BATCH_SIZE, SEQ_LEN, N))

    benchmark(scan, deltaA, deltaB_u, C)


@pytest.mark.benchmark(warmup=False)
def test_parallel_scan(benchmark):
    mx.random.seed(0)
    D, N = 768, 16

    deltaA = mx.random.normal((BATCH_SIZE, SEQ_LEN, D, N))
    deltaB_u = mx.random.normal((BATCH_SIZE, SEQ_LEN, D, N))
    C = mx.random.normal((BATCH_SIZE, SEQ_LEN, N))

    y = benchmark(parallel_scan, deltaA, deltaB_u, C)
    yref = scan(deltaA, deltaB_u, C)

    # print(y)
    # print(yref)
    max_diff = (y - yref).abs().max().item()

    assert allclose(y, yref), f'max_diff = {max_diff}'
