# SiameseConvNet
Performs cognate identification using Siamese Convolutional Networks

##Please consider citing the following paper if you use the code:
Taraka Rama. Siamese Convolutional Networks for Cognate Identification. Proceedings of COLING 2016, Osaka, Japan, 2016.	http://aclweb.org/anthology/C/C16/C16-1097.pdf

##Requirements:
  - Need Keras (https://keras.io/) with Tensorflow (https://www.tensorflow.org/) as  a backend for running the code.
  
##Running the program:
 - Run a program as ```python siamese_cognates_cnn_langs_info.py 30 data/IELex-2016.tsv.asjp```
 - The program takes the number of concepts for training and the name of the training dataset as commandline arguments
 - There are total four programs. 
 - The program starting with one_hot uses the 1-hot representation of a phoneme to represent a word as a matrix and then compute the similarity between the two words and then optimizes binary cross-entropy using a Adadelta optimizer
 - The program starting with siamese_cognates uses articulatory features for representing a phoneme and then performs convolutional operations for identifying cognates
 
 - The program outputs the F-scores and accuracies for a dataset to the screen
 - The three datasets used in the experiments are provided in data folder
  
##Contact:
In case of any questions, contact taraka-rama.kasicheyanula@uni-tuebingen.de
