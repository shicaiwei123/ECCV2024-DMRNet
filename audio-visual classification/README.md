





## Usage
### Data Preparation
Download Dataset：
[CREMA-D](https://pan.baidu.com/s/1bHpGxvjCDQkfgMXD_fhEdg?pwd=w36h), [Kinetics-Sounds](https://pan.baidu.com/s/1E9E7h1s5NfPYFXLa1INUJQ?pwd=rcts).
Here we provide the processed dataset directly. T

The original dataset can be seen in the following links,
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D),
[Kinetics-Sounds](https://github.com/cvdfoundation/kinetics-dataset). And you need to process the dataset following the instruction below.


### Pre-processing

For CREMA-D and VGGSound dataset, we provide code to pre-process videos into RGB frames and audio wav files in the directory ```data/```.

#### CREMA-D 

As the original CREMA-D dataset has provided the original audio and video files, we simply extract the video frames by running the code:

```python data/CREMAD/video_preprecessing.py```

Note that, the relevant path/dir should be changed according your own env.  


### data path

you should move the download dataset into the folder data


## Train 

We provide bash file for a quick start.

For CREMA-D

```bash
bash cramed.sh
```


## Test

```python
python valid.py
```

## Comparison

We also provide the code for MMANet and ShaSpec, you can conduct experiments on the comparison methods with the bash files.

For MMANet
```bash
bash cramed_mmanet.sh

```

For ShaSpec
```bash
bash cramed_shaspec.sh

```
And you can also test these methods with the functions in valid.py after training.
