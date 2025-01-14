# Install on MacOS

## Arm64

**Conda:** create a separate conda environment for PyTorch (and do not activate it)

```
conda create -n pytorch_swift pytorch
```

Check out and update the MOOSE submodule

```
git submodule update --init moose
```

Use the MOOSE configure script to enable libtorch and point it to the pytorch conda instalaltion.

```
cd moose
export LIBTORCH_DIR=$(conda activate pytorch_swift; echo $CONDA_PREFIX; conda deactivate)
./configure --with-libtorch
```

Step back into the swift root directory and build.

```
cd ..
make -j
```

Swift on arm64 macs installed this way will support `cpu` and `mps` as compute devices. An mps/cpu (12 threads) runtime comparison for an M2 max laptop is shown below. Note that MPS only supports single precision (32bit) floating point numbers.

| Example | `cpu` runtime | `mps` runtime |
| - | - | - |
| `cahnhilliard.i` | 128s | 60s|
| `cahnhilliard2.i` | ~26000s | 1068s |
| `cahnhilliard3.i` | 440s | 29s |
| `rotating_grain.i`| 54s | 21s |
