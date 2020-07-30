# Weakly-Supervised Classification of HER2 Expression in Breast Cancer Haematoxylin and Eosin Stained Slides
## About
Implementation of the paper [_"Weakly-Supervised Classification of HER2 Expression in Breast Cancer Haematoxylin and Eosin Stained Slides"_](https://www.mdpi.com/2076-3417/10/14/4728), by Sara P. Oliveira, João Ribeiro Pinto, Tiago Gonçalves, Rita Canas-Marques, Maria J. Cardoso, Hélder P. Oliveira and Jaime S. Cardoso.
## Clone this repository
To clone this repository, open a Terminal window and type:
```bash
$ git clone https://github.com/spoliveira/HERclassHE.git
```
Then go to the repository's main directory:
```bash
$ cd HERclassHE
```
## Dependencies
Please be sure that you have the following Python packages installed:
```
Package Name             Version            

cudatoolkit               9.2               
ghalton                   0.6.1                           
numpy                     1.16.3           
opencv                    4.1.0          
openslide                 3.4.1              
openslide-python          1.1.1              
pillow                    5.4.1        
python                    3.7.2           
pytorch                   1.0.1           
scikit-image              0.15.0           
scikit-learn              0.21.1          
scipy                     1.2.1            
torchvision               0.2.1
```
Or you can create a virtual Python environment (Python 3.7) and run:
```bash
$ pip install -r requirements.txt
```
## Data
If you need help with access to the data used in this paper, please send an e-mail to: [sara.i.oliveira@inesctec.pt](mailto:sara.i.oliveira@inesctec.pt).
## Usage
We advise you to read the [full paper](https://www.mdpi.com/2076-3417/10/14/4728) to understand which task may fit your purposes or if you want to replicate our work.
### 1. Data Pre-processing
#### 1.1 Inference Only (HE Images)
To perform inference-only on HE images you need to go to the ["preprocessing"](preprocessing/) directory:
```bash
$ cd preprocessing
```
Open ["HASHI_segmentation.py"](preprocessing/HASHI_segmentation.py) on your IDE, and edit the following variables:
```python
# --- Data directories ---
dir_slides = ''                 # path with all the slides, in this case, you should put here the path to HE images slides
dir_thumbnails = ''             # path where segmentation masks will be stored (reduced size)
```
This file is used to generate tumor segmentation masks. To run this script, type:
```bash
$ python HASHI_segmentation.py
```
Then, open ["wsi_main.py"](preprocessing/wsi_main.py) on your IDE, and edit the following variables:
```python
# --- Data directories ---
dir_slides = ''         # path with all the slides, in this case, you should put here the path to HE images slides
dir_patches = ''        # path where .pkl files will be stored (one file per slide containing [patches, coord, label(IHC score), bin_label(IHC status)])
dir_thumbnails = ''     # path where thumbnails will be stored (img/otsu masks)
dir_masks = ''          # annotation masks location (if there are annotation mask files)
labels_file = ''        # .csv file with labels (structure: slide_name, IHC score (4 classes), IHC status (2 classes))
```
This file is used to generate the _.pickle_ files that contain the tiles that will be used. To run this script, type:
```bash
$ python wsi_main.py
```
#### 1.2 Training (HER2/HE Images)
