# Consumer tracking using ByteTrack + LSTM

Tracking code for the experiments of the paper [A New Benchmark for Consumer Visual Tracking and Apparent Demographic Estimation from RGB and Thermal Images](https://www.mdpi.com/1424-8220/23/23/9510), published in Sensors, 2023.

## Installation instructions

Before starting, update g++.

Clone the repository in a directory of your choice and change to that directory and then to "Consumer-tracking".

Create a virtual environment with conda, we used Python 3.7.16 for all experiments:

```
conda create --name ENV python=3.7
conda activate ENV
```

Install the code requirements, enter the commands in the order presented:
```shell
pip install numpy
pip install cython
pip install lap
pip install -r requirements.txt
```

Build YOLOX by running:
```shell
python setup.py develop
```

Install [pycocotools](https://github.com/cocodataset/cocoapi):

```shell
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Install the remaining packages:
```shell
pip install cython_bbox
```
>If open-cv fails to install (error importing cv2), use the following command: ```pip install opencv-python-headless```.

Finally, install scikit-learn (for linear assignment):
```shell
pip install scikit-learn==0.22.2
```

## Data preparation

1. Download the _Consumers_ [dataset](https://www.kaggle.com/datasets/angelosgiotis/consumers-bid/) from Kaggle.
2. Unzip and place the files under ```datasets``` directory or create a symbolic link inside that directory using ```ln -s``` if you place them somewhere else.

## Training

>**Optional, you can skip this process and use our [checkpoints](#checkpoints).**

### Training YOLOX on the mixed dataset

To train on the mixed dataset (MOT, CrowdHuman, ETH, CityPersons), follow the [instructions](https://github.com/ifzhang/ByteTrack#data-preparation) to prepare the data.
>Alternatively, download the ByteTrack pretrained weights.

### Training YOLOX on the Consumers anonymized dataset

Use the ByteTrack [checkpoints](https://github.com/ifzhang/ByteTrack#model-zoo) as starting weights.

>To train the **detectors** on the Consumers dataset you need to prepare it to COCO format for YOLOX and then create *experiment* files for each backbone instance. Refer to [ByteTrack](https://github.com/ifzhang/ByteTrack#training) for the steps of this process.
Each file needs the dataset path in order to set up data loading.

### Training the LSTM on MOT17

To train the LSTM models, follow the instructions from [DEFT](https://github.com/MedChaabane/DEFT).

## Checkpoints

All checkpoins (detector weights, LSTM weights) from our experiments can be downloaded from [here](https://drive.google.com/drive/folders/1v4LP830BAH_rF8YjC6UmlCTlAILihAui) [Google Drive].

### Tracking results on _Consumers_ (anonymized) test set (with LSTM)

| Model   | Size | MOTA | MOTP | IDF1 |   |
| -----   | ---- | ---- | ---- | ---- | - |
| YOLOX-X | 797M | 81.5 | 90.2 | 85.6 |   |
| YOLOX-L | 414M | 80.9 | 89.3 | 87.6 |   |
| YOLOX-M | 194M | 82.5 | 88.9 | 86.3 |   |
| YOLOX-S |  69M | 82.9 | 88.3 | 87.1 |   |
| YOLOX-T |  39M | 81.1 | 88.4 | 87.3 |   |
| YOLOX-N |   7M | 81.3 | 86.8 | 87.5 |   |

Place the checkpoints inside _weights_ directory.
>If you change the name or path you will need to specify it with an argument when running the [evaluation](#evaluation-on-consumers-anonymized-test-set) script.

>It is important to acknowledge that the inferred results on the published evaluation datasets for the proposed (ByteTrack+LSTM) method may exhibit slight variations compared to the reference performance indices in Tables 2 and 3 (training on the raw Consumers train set and inferring on the 'defaced' evaluation set). This marginal performance decrease can be attributed to the manual post-anonymization step for instances that were inadvertently missed by the defacement software.

### Re-ID models

Some tracking methods require re-id models to function, if you evaluate methods other than byte, we also provide those in the above link.
Download and place the weights inside ```weights``` directory.

## Evaluation on Consumers anonymized test set

We created an evaluation script for easy experimentation:

For example, to use YOLOX-M, run:
```shell
python tools/evaluate.py -f exps/yolox_m.py -c weights/yolox_m.pth.tar --fp16 --fuse --track_thresh 0.3 --conf 0.1
```

Additional arguments can be used to modify the tracking parameters:

```
-f, --exp_file                 specify experiment file
-b, --benchmark                measure fps only, skip tracking results
--path, --data-path            specify path to dataset
--expn, --experiment-name      specify a directory name to save results in
-c, --ckpt                     specify weights file for detector
--conf                         threshold for detection confidence
--nms                          threshold for non-maximum-supression
--track_thresh                 tracking threshold for boxes
--match_thresh                 matching threshold for linear assignment
--aspect_ratio_thresh          filters boxes with high aspect ratio
--min_box_area                 filters tiny boxes

--lstm                         use LSTM instead of Kalman for motion prediction (default = True)
--lstm_layers                  specify layers in the LSTM (default = 1)
--lstm_neurons                 specify neurons per layer in the LSTM (default = 128)
--lstm_weights                 specify path to LSTM weights (default = weights/)
--method                       select tracking method (byte, sort, deepsort, motdt, strongsort)
```
>By trying different values of the tracking hyperparameters (e.g. `track_thresh`) you can get different (higher or lower) results compared to the previous table, or you can set different thresholds per sequence (check [this example](https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py#L133) on how to do this).

>For OC-SORT evaluation, follow the instructions on their [repository](https://github.com/noahcao/OC_SORT/tree/2d34c67d58f89b0762ce912c38f94746175e2903), and copy the _exp_ files to that directory. The process is similar.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Acknowledgements

Modifications built on top of the tracking codebase from [ByteTrack](https://github.com/ifzhang/ByteTrack).

LSTM implementation from [DEFT](https://github.com/MedChaabane/DEFT).

YOLOX code from the official [repo](https://github.com/Megvii-BaseDetection/YOLOX).

## Citation
If you use this codebase or the _Consumers_ dataset in your work please cite:

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