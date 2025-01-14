# Install MacOS

## Arm64

**Conda:** create a separate conda environment for pytorch (and do noyt activate it)

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
