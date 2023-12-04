# Apparent attribute (age group, gender) estimation using Inception + ALM module

Apparent attribute estimation code for the experiments of the paper [A New Benchmark for Consumer Visual Tracking and Apparent Demographic Estimation from RGB and Thermal Images](https://www.mdpi.com/1424-8220/23/23/9510), published in Sensors, 2023.

## Installation instructions

After cloning the repository and changing to `Apparent-age-gender-estimation` directory, create a virtual environment with conda, we used Python 3.7.16 for all experiments:

```
conda create --name ENV python=3.7
conda activate ENV
```

> Alternatively, use the ENV from [Consumer-tracking](https://github.com/jpanagos/consumers-bid/tree/master/Consumer-tracking).

Install the code requirement packages:
```
pip install torchvision
```
This should also install all requirements (numpy, torch, cuda, Pillow).

## Dataset

Download the _BID_ dataset from [Kaggle](https://www.kaggle.com/datasets/angelosgiotis/consumers-bid) and place the folders under the `datasets/BID/` directory.

> If the files are placed elsewhere, edit the `utils/datasets.py` file, lines #204 and #206 with the appropriate path.

We provide the lists used for training and testing, under the `data_list/BID/` directory.

## Training and testing (anonymized _BID_)

First download the pre-trained inception [weights](http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth) and place inside `model` directory:
```
wget http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth --no-check-certificate
mv bn_inception-52deb4733.pth model/
```
Then, to train the model with default settings:
```
python main.py
```

Training can be parameterized with the following arguments:
```
--epochs                        specify amount of epochs
--lr, --learning-rate           set learning rate (default = 0.0001)
--batch_size                    set batch size (default = 32)
--optimizer                     set optimizer (default = 'adam')
--momentum                      specify momentum amount (default = 0.9)
--weight_decay                  specify weight decay amount (default = 0.0005)
--decay_epoch                   specify epochs to decay the learning rate (default = (20,40))
--workers                       set number of workers for data loading
--att, --attention              specify type of attention on ALM 
--act, --activation             specify type of activation function (for SB attention only)

--resume                        path to checkpoint file (for fine-tuning or evaluating)
--exp                           path to save experiments to
```

## Evaluating on the anonymized _BID_ test set

### Checkpoints

We provide [checkpoints](https://drive.google.com/drive/folders/1uK2eG1v8z8al4ifpmrkYspf9jxqO9vYJ?usp=drive_link) from our experiments (trained with **raw** data):

[Inception, ALM, SE, 16](https://drive.google.com/file/d/1u-NKMddw3cHQH1Opul7Fl2wAPBMaAp9v/view?usp=sharing) - 43.7MB

For evaluation, run the following command:
```
python main.py -e --exp evaluation --resume <path_to_weights>
```

| Method | mA | Acc | Prec | Rec | F1 |
| ------ | -- | --- | ---- | --- | -- |
| Inception w/ ALM (SE attention, ratio=16) | 74.8% | 61.9% | 65.0% | 73.9% | 69.2% |

>It is important to acknowledge that the inferred results on the published evaluation datasets for the proposed ALM method may exhibit slight variations compared to the reference performance indices in Table 6 (training on the raw BID train set and inferring on the 'defaced' evaluation set). This marginal performance decrease (approximately 1-2%) can be attributed to the manual post-anonymization step for instances that were inadvertently missed by the defacement software.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Acknowledgements

Modifications built on top of the codebase of [ALM](https://github.com/chufengt/ALM-pedestrian-attribute).

## Citation
If you use this codebase or the _BID_ dataset in your work please cite:

```
@Article{s23239510,
AUTHOR = {Panagos, Iason-Ioannis and Giotis, Angelos P. and Sofianopoulos, Sokratis and Nikou, Christophoros},
TITLE = {A New Benchmark for Consumer Visual Tracking and Apparent Demographic Estimation from RGB and Thermal Images},
JOURNAL = {Sensors},
VOLUME = {23},
YEAR = {2023},
NUMBER = {23},
ARTICLE-NUMBER = {9510},
URL = {https://www.mdpi.com/1424-8220/23/23/9510},
ISSN = {1424-8220},
DOI = {10.3390/s23239510}
}
```
