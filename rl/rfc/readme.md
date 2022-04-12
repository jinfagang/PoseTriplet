# Residual Force Control

original repo from: https://github.com/Khrylx/RFC


## Install

I am trying port to new `mujoco` python bindings, but currently use the old one. First download mojoco 21: https://github.com/openai/mujoco-py#install-mujoco

Then install `libglew-dev`, also add some path to LD_LIBRARY_PATH.

Rember run: `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so` before runing.


## Demo

For quick demo, run:

```
python vis_im.py --cfg 0506 --iter 1000
```

`0506` means the clip id of mocap to imitate.


## Train

to training, we need first convert the mocap data first:

```
python convert_cmu_mocap.py --amc_id 05_06 --out_id 05_06_test --render
```

remove `--render` will direclty gen a file `data/cmu_mocap/motion/05_06_test.p`.

then train:

```
export OMP_NUM_THREADS=1
python motion_im.py --cfg 0506 --num_threads 1 --render
```
seems training very stucked.

