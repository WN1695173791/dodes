# small ddpmg

```shell
python main.py --config configs/vp/ddpm/cifar10_continuous.py --mode sampling --ckpt ckpts/cifar10_ddpm_continuous_l --result_folder logs/deleteme --config.eval.num_samples=2000 --config.sampling.method=ei
```


```shell
python main.py --config configs/vp/ddpm/cifar10_continuous.py --mode sampling --ckpt ckpts/cifar10_ddpm_continuous_l --result_folder logs/ei_4_10 --config.eval.num_samples=50000 --config.sampling.method=ei
```