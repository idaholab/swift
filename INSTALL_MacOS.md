# Install on MacOS

## Arm64

**Conda:** create a separate conda environment for pytorch (and do not activate it)

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
./configure --with-libtorch=$CONDA_ENV_DIR/pytorch_swift
```

Step back into the swift root directory and build.

```
cd ..
make -j
```

Swift on arm64 macs installed this way will support `cpu` and `mps` as compute devices. On an M2 max laptop the 2D Cahn-Hilliard example will run in ~60s on mps and ~130s on cpu (12 threads).
