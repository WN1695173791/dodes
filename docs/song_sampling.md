# play ipynb

## install jupyter

```shell

# node js
curl -fsSL https://deb.nodesource.com/setup_17.x | sudo -E bash -
sudo apt-get install -y nodejs

pip install -U jupyterlab
pip install -U ipympl
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib
pip install jupyterlab_execute_time
pip install jupyterlab_vim
pip install jupyterlab-lsp
pip install 'python-lsp-server[all]'
jupyter labextension update --all
jupyter lab build
```

## download ckpts

```shell
# vp/cifar_ddpm_continuous
mkdir -p ckpts
gdown 1bLkJGwgX1kPlMdYv1TDtTRLecpUCokON -O ckpts/cifar10_ddpm_continuous_l

gdown 1jFmheW6vFKUzvPCW2uCgaUskrGcomj8u -O ckpts/cifar10_ncsnpp_continuous
```

## check `sampling.ipynb`


# run fid

```shell
python main.py --config configs/vp/ddpm/cifar10_continuous.py --mode sampling --ckpt ckpts/cifar10_ddpm --result_folder logs/deleteme --config.eval.num_samples=3000 --config.sampling.method=ode
```