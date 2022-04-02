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
mkdir -p ckpts/

# vp/cifar_ddpm_continuous
gdown 1bLkJGwgX1kPlMdYv1TDtTRLecpUCokON -O ckpts/cifar10_ddpm
```

## check `sampling.ipynb`