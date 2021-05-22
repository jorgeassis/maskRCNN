
## Instal miniconda

https://docs.conda.io/en/latest/miniconda.html#linux-installers

## Create a VE

conda install python=3.7.1

conda create -n "mrcnn" python=3.7.1
// conda create -n "mrcnn" python=3.7.1 ipython

conda info --envs
conda activate mrcnn

/home/myname/miniconda3/envs/mrcnn/bin/python3

## Export environment from previous PC

pip freeze > requirements.txt
pip install -r requirements.txt

conda env export > requirements.yml
conda env create -f requirements.yml
conda-env create -n mrcnn -f=requirements.yml

## Installation
0. Python 3.7.10
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)