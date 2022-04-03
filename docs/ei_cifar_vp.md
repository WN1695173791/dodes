## checkponit 

```shell
gdown 1-QDUNpQaP85h7KuyC-QKCYQ0ZOTg9ke- -O ckpts/cifar10_ddpmpp_deep_continuous_l
gdown 1XovuGyunrF7mQtTd7LZY3ud1eTvaZgW2 -O ckpts/cifar10_ddpmpp_deep_continuous_h

gdown 1omSz_KJcLShqagrhTm7g0XTITzXb9qqu -O ckpts/cifar10_ddpmpp_continuous_l
gdown 1A9u9CrvUbxho6j3TjxVOtc-K9n701seT -O ckpts/cifar10_ddpmpp_continuous_h

gdown 1BQNnfEckADGLF9Vny5jq18cAVK9weADj -O ckpts/cifar10_ddpmpp
```

## checkit

```shell
python() {
    echo $@
}
for order in 1 4;
do
    for step in 5 10 20;
    do
        for ckpt in cifar10_ddpmpp_continuous_l cifar10_ddpmpp_continuous_h;
        do
            python main.py --config configs/vp/cifar10_ddpmpp_deep_continuous.py --mode sampling --ckpt ckpts/$ckpt --result_folder logs/ei_$ckpt\_$order\_$step --config.sampling.method=ei --config.sampling.ei_step=$step --config.sampling.ei_order=$order
        done

        for ckpt in cifar10_ddpmpp_continuous_l cifar10_ddpmpp_continuous_h;
        do
            python main.py --config configs/vp/cifar10_ddpmpp_continuous.py --mode sampling --ckpt ckpts/$ckpt --result_folder logs/ei_$ckpt\_$order\_$step --config.sampling.method=ei --config.sampling.ei_step=$step --config.sampling.ei_order=$order
        done

        for ckpt in cifar10_ddpmpp;
        do
            python main.py --config configs/vp/cifar10_ddpmpp_continuous.py --mode sampling --ckpt ckpts/$ckpt --result_folder logs/ei_$ckpt\_$order\_$step --config.sampling.method=ei --config.sampling.ei_step=$step --config.sampling.ei_order=$order
        done
    done
done
```