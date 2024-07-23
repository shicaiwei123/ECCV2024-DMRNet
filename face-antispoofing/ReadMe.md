



## dataset preparation
download link: [CAISIA-SURF](https://pan.baidu.com/s/1h_vVQof0cdkLd396sVi1Rg?pwd=ao07)

build soft link:
``` bash
    cd face-antispoofing/data
    ln -s /home/ssd_data/liveness/CASIA-SURF/  CASIA-SURF
```
you need convert the path to your own data

## run

```bash
    cd face-antispoofing/src
    bash  surf_dulconv_kl_auxi_share_0.1.sh
```

## test

```bash
    cd face-antispoofing/test
    python general_missing_dulconv.py 0 0 0 0 0
    
```


## Comparison

We also provide the code for MMANet and ShaSpec, you can conduct experiments on the comparison methods with the bash files.

For MMANet

see 

For ShaSpec
```bash
bash surf_shaspec.sh

```
And you can also test these methods with the functions in valid.py after training.
