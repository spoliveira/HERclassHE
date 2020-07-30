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
If you need help with access to the data used in this paper, please send an e-mail to: [sara.i.oliveira@inesctec.pt](mailto:sara.i.oliveira@inesctec.pt)