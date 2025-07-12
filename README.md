# NeRF-based Action Evaluation for Robotic Grasping

This is an end-to-end imitation-learning algorithm that builds an instant-NeRF to synthesize novel views for 6-DoF pick and place affordance prediction and solves various 6-DoF tabletop tasks.

> Reproduced and modified [MIRA](https://arxiv.org/abs/2212.06088).

## Guides

- Setup: [Installation](#installation)
- Quickstart: [Dataset Generation](#dataset-generation), [Training](#training), [Evaluation](#evaluation)

## Installation

**Requirements**

- CUDA 11
- CuDNN 8

> We highly recommend CUDA 11.8.0 and CuDNN 8.9.4

**Step 1.** Create and activate Conda environment, then install GCC and Python packages.

```shell
conda create --name NAERG python=3.7 -y
conda activate NAERG
sudo apt-get update
sudo apt-get -y install gcc libgl1-mesa-dev
git clone git@github.com:Wushr-Lance/NeRF-based-Action-Evaluation-for-Robotic-Grasping.git
cd ~/NeRF-based-Action-Evaluation-for-Robotic-Grasping
pip install -r requirements.txt
python setup.py install --user
```

**Step 2.** Install [orthographic-ngp](https://github.com/yenchenlin/orthographic-ngp/) and its dependencies.
```shell
git clone --recursive git@github.com:yenchenlin/orthographic-ngp.git
cd orthographic-ngp
sudo apt-get install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev \
                     libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev
sudo apt install cmake
cmake . -B build
cmake --build build --config RelWithDebInfo -j 16
cd ../
```

**Step 3.** Other preparations.
```shell
pip install protobuf==3.20.*
pip install pyexr
pip install tqdm
conda install -c conda-forge libstdcxx-ng
sudo prime-select nvidia # Then reboot your PC! <-- IMPORTANT!
```

## Dataset Generation
Generate training and test data (saved locally). Note: remove `--disp` for headless mode.

```shell
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=$TASK --mode=train --n=10
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=$TASK --mode=test --n=10
```

where `$TASK` can be `block-insertion-sixdof`, `place-red-in-green-sixdof`, `hanging-disks`, `stacking-kits`, etc.

## Training

Train a model. Model checkpoints are saved to the `checkpoints` directory.

```shell
python ravens/train.py --task=$TASK --agent=mira --n_demos=10 --gpu_limit 12
```

## Evaluation

Evaluate a NeRF-Porter using the model trained for 5000 iterations. Results are saved locally into `.pkl` files.


```shell
python ravens/test.py --assets_root=./ravens/environments/assets/ --disp=True --task=$TASK --agent=mira --n_demos=10 --n_steps=5000
```

## Acknowledgement

This codebase is highly based on the following publication:

```
@article{zeng2020transporter,
    title={Transporter Networks: Rearranging the Visual World for Robotic Manipulation},
    author={Zeng, Andy and Florence, Pete and Tompson, Jonathan and Welker, Stefan and Chien, Jonathan and Attarian, Maria and Armstrong, Travis and Krasin, Ivan and Duong, Dan and Sindhwani, Vikas and Lee, Johnny},
    journal={Conference on Robot Learning (CoRL)},
    year={2020}
}
```
