# LS-IQ: Implicit Reward Regularization for Inverse Reinforcement Learning
This is the official code base of the paper [*LS-IQ: Implicit Reward Regularization for Inverse Reinforcement Learning*](https://arxiv.org/pdf/2303.00599.pdf), 
which was presented at the eleventh International Conference on Learning Representations ([ICLR 2023](https://iclr.cc/Conferences/2023))
in Kigali Ruanda. Here, we also provide all the baselines for the [LocoMuJoCo](https://github.com/robfiras/loco-mujoco) imitation learning benchmark [*LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion*](https://arxiv.org/pdf/2311.02496.pdf) presented at the Robot Learning workshop at [NeurIPS 2023](https://nips.cc/).

---
![Divergence_Minimization](img/Divergence_Minimization.gif)
## Method
Within this work, we analyze the effect of a squared norm regularizer on the implicit reward function in the inverse reinforcement learning setting.
We build on previous work ([IQ-Learn](https://arxiv.org/pdf/2106.12142.pdf)), and show that this regularizer results in a minimzation
of the Chi^2-Divergence between the expert and a mixture distribution. **We show that - unlike previously used divergences - this divergence is bounded 
and the resulting reward function is also bounded**. An example is given in the picture above, where the target distribution is blue, 
the current policy distribution is green, and the mixture is orange. As can be seen, the vanilla Chi^2 divergence can reach very high values - despite the support area being non-zero -
while the divergence on the mixture is bounded. Both optimization share the same optimal solution.

Also, this regularizer provides a particularly illuminating perspective: the original **objective can be understood as
squared Bellman error minimization with fixed rewards for the expert and the policy**. This setting can be further used to
stabilize training as shown in our paper.

### Key Advatanages 
✅ Simple implementation on top of SAC \
✅ Bounded objective with bounded reward yields stable and convenient training\
✅ Retains performance even without expert actions\
✅ Performs even when only 1 expert trajectory is given\
✅ Works in complex and realistic environments such as on the Atlas Locomotion task\
✅ Unlike previous methods, no survival bias!

---
## Installation
You can install this repo by cloning and then 

```shell
cd ls-iq
pip install -e .
```

### Download the Datasets [not needed for LocoMuJoCo]
In order to run the examples and reproduce the results, you have to download the datasets used in our paper. To do so, you have to install `gdown`:

```shell
pip install gdown
```
Then you can just run the download script:
```shell
chmod u+x ./download_data.sh
./download_data.sh
```

---
## Examples
You can find launcher files in the example folder to launch all different versions of LSIQ and to reproduce the main results
of the paper. 

Here is how you run the training of LSIQ with 5 expert trajectories on all Mujoco Gym Tasks:

```shell
cd examples/02_episode_5/
python launcher.py
```
To monitor the training, you have to use Tensorboard. Once the training is launched, the directory `logs` will be created, which contains
the Tensorboard logging data. Here is how you run Tensorboard:

```shell
tensorboard --logdir logs
```


Some experiments were such as the Atlas locomotion task were conducted on environment, which are yet not
available on Mushroom-RL, but will be available soon! Once the environments are part of Mushroom-RL, the experiment files will be added here.
Follow Mushroom-RL on Twitter [@Mushroom_RL](https://twitter.com/Mushroom_RL) to immediately get notified once the
new environment package is available!

---
## Citation
```
@inproceedings{alhafez2023,
title={LS-IQ: Implicit Reward Regularization for Inverse Reinforcement Learning},
author={Firas Al-Hafez and Davide Tateo and Oleg Arenz and Guoping Zhao and Jan Peters},
booktitle={Eleventh International Conference on Learning Representations (ICLR)},
year={2023},
url={https://openreview.net/pdf?id=o3Q4m8jg4BR}}
```
