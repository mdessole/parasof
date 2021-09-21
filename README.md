# PARASOF
Pallel GPU solver for Bordered Almost Block Diagonal (BABD) matrices. Based on

> Monica Dessole, Fabio Marcuzzi "**[A massively-parallel algorithm for Bordered Almost Block Diagonal systems on GPUs](https://link.springer.com/article/10.1007/s11075-020-00931-8)**", Numerical Algorithms  86, 1243–1263 (2021).

## Running parasof

Clean existing shared library
```console
rm src/*.so
```

In order to run a test one shoud execute the file `test.py` with the inputs `N` (number of internal block rows), `n` (block size), `N_r` (number of internal block rows of the reduced system). For example
```console
python test.py 9 8 3
```
solves a BABD system with (N+1) block rows with square blocks of size 8x2 with PARASOF.

## Dependencies

- Pyrhon 3 or >
- CUDA 10
- Pucuda
- numpy
- scipy
