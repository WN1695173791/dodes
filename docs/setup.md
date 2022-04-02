## Docker env

```
[qinsheng/cudot](https://hub.docker.com/layers/qinsheng/cudot/edge/images/sha256-db3afe1e0f4d8f844b0afaaa75a217b59341ee28a7ff5d1a89368a72a8ab735b?context=explore)
```

## installation

```shell
pip install -r requirements.txt
pip install --upgrade jax==0.2.8 jaxlib==0.1.59+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install flax==0.3.1
pip install keras==2.6.0
pip install tensorflow-probability==0.13.0
pip install --upgrade tensorflow-estimator==2.6.0
pip install einops
```

### setup env

```shell
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-11.2"
echo 'export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-11.2"' >> $HOME/dotfiles/zsh/machine_local_after
```

## cifar10_stats

```shell
pip install gdown
mkdir -p assets/stats
cd assets/stats
gdown --id 1fXgBupLzThTGLLsiYCHRQJixuDsR1bSI
```

## train simple cifar

```shell
# save to setup
python main.py --config configs/vp/ddpm/cifar10_continuous.py --mode train --workdir logs/delete_me
```