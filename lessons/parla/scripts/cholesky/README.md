# Gradual Adoption Example: Cholesky Factorization

Parla does not require users to structure their applications to use specific data structures or classes to use its parallelism. Instead, Parla provides a set of annotations that can be added to sequential code to gradually parallelize an application.  More advanced features can be added slowly such as async semantics, placement constraints, function variants, and Parla arrays.

This example demonstrates how to use Parla's annotations to parallelize a [Block Left-Looking Lower Cholesky factorization algorithm](https://www.netlib.org/utk/papers/factor/node9.html) in stages from a sequential CPU implementation to a hybrid CPU/GPU implementation.  

- [Sequential CPU Implementation](01_cholesky_serial.py)
- [Parallel CPU Implementation with Parla](02_cholesky_cpu.py)
- [Parallel CPU/GPU Implementation with Parla](03_cholesky_manual.py)
- [Parallel CPU/GPU Implementation with Parla Arrays](04_cholesky_automatic.py)

## Example Description

The original sequential algorithm loops over the blocks of the matrix. For each diagonal block, it performs a smaller subfactorization on the diagonal block and then performs a series of updates on the remaining blocks. 

For a small example of how to add Parla annotations consider the following snippet that performs the symmetric updates from the column onto the remaining diagonal blocks. 

```python
for j in range(n_blocks):
	for k in range(j):
		A[j][j] = syrk(A[j][k], A[j][k], A[j][j])
```

This snippet can be parallelized by adding a `@spawn` decorator to launch inline tasks for each iteration. 
At the moment, dependencies in Parla are explicit and not inferred from data access. 
From the structure of the algorithm we know that the updates to successor diagonal blocks depend on the prior panel factorization of $j, k$ at each step. 


To avoid concurrent writes, we explicitly add a dependency on prior update tasks to the same block. An `exclusive` constraint to loosen this dependency is currently not supported.

```python
for j in range(n_blocks):
	for k in range(j):
        @spawn(SYRK[j, k], [TRSM[j, k], SYRK[j, 0:k]])
		def t1():
			A[j][j] = syrk(A[j][k], A[j][k], A[j][j])
```

In `03_cholesky_manual.py` we demonstrate how to use Parla's placement constraints to specify that the `SYRK` tasks should be placed on the GPU.

Here (if the API for the operation was different between CPU and GPU implementations), we could use function variants to specialize a GPU implementation of syrk. 
See the example for examples of this for `trsm` and `potrf`. 

```python
@specialize
def gemm(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    C -= A @ B.T
    return C

@gemm.variant(gpu)
def gemm_gpu(A: cp.ndarray, B: cp.ndarray, C: cp.ndarray) -> cp.ndarray:
    C -= A @ B.T
    return C

for j in range(n_blocks):
    for k in range(j):
        @spawn(SYRK[j, k],[TRSM[j, k], SYRK[j, 0:k]], 
            placement=gpu)
        def t1():
            A_jj = clone_here(A[j][j])
            A_jk = clone_here(A[j][k])
            A[j][j] = syrk(A_jk, A_jk, A_jj)
```

`clone_here` annotations are added to bring in the data to the current device if it is not already present.

In `04_cholesky_automatic.py` we demonstrate how to use Parla arrays to automatically manage data placement and cloning. This also allows the runtime to use data access information for scheduling and task placement decisions. 

## Running the Examples

```bash 
# Run the sequential CPU implementation
./.../run.sh 00_build_matrix.py --size 10000 
./.../run.sh 01_cholesky_serial.py --input matrix.npy --blocks 10

```