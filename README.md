# Machine Learning Approach Towards Emotion Recognition from EEG Recordings

###   ECE-GY 6123 "Introduction to Machine Learning" course project

## Overview
#### Please check [project report](https://github.com/vbabushkin/ECE-GY6123_ML_PROJECT/blob/main/REPORT/mlProjectReport.pdf) for more details

<div align="justify">
Classifying human emotional responses from bio-sensing signals is getting more interest in applications for Human Computer Interaction. Traditionally most of the studies focus on human emotion recognition from visual stimuli. Advancements in haptics made it possible to study emotions elicited by tactile modality. In this work, we address the emotion classification from the EEG recordings of neural activations elicited by visual stimuli. The aim is to establish foundations for interpreting the emotional response from the neural activations in the presence and absence of tactile simulation by relating it to the observed visual cue which elicited emotional response is already known.  
</div>  
<br>
<div align="justify">
We present a comparative analysis of Support Vector Machine (SVM) and Convolutional Neural Network (CNN) for supervised classification of emotional states from the EEG recordings of neural activations evoked by the visual cues from International Affective Picture System (IAPS) database . CNN requires little or no preprocessing of the data and is capable of inferring the hidden dependencies in the data. In contrast, the SVM classifier is faster to train and easier to implement. We used 4, 9, and 12 emotional subcategories defined from the ranking of the subjects collected during the experiment. The instances in a dataset of 4 emotional categories were equally distributed in 4 classes, but datasets with 9 and 12 categories contained heavily underrepresented classes. Both SVM and CNN classifiers performed well on the unbiased dataset, achieving average accuracies around 0.85 for SVM and 0.81 for CNN. However increasing the number of classes also introduces a bias in the number of trials, the accuracy drops. In this case, both classifiers achieve accuracies of 0.70 and 0.69 for 9 and 12 classes correspondingly (see 
Fig. 1 and Fig. 2).  
</div> 
<br>
<div align="justify">
Despite heavily biased datasets in the case of 9 and 12 classes, the SVM classifier performed better than CNN, demonstrating high robustness to the classification of misrepresented instances. In the future SVM, combined with visual cues for emotional response labeling, can be used for multimodal classification of emotional states elicited by the presence and absence of tactile stimulation.
</div> 

<p>
<img  src="https://github.com/vbabushkin/ECE-GY6123_ML_PROJECT/blob/main/FIGURES/accuracy_svm_10fold_all_classes.png"  alt="svm-accuracy"/>
<br>
<em>Fig. 1: The accuracy of SVM classifier for 4, 9 and 12 classes after 10-fold cross validation.</em>
</p>


<p>
<img  src="https://github.com/vbabushkin/ECE-GY6123_ML_PROJECT/blob/main/FIGURES/accuracy_cnn_10fold_all_classes.png"  alt="cnn-accuracy"/>
<br>
<em>Fig. 2: The accuracy of CNN classifier for 4, 9 and 12 classes after 10-fold cross validation.</em>
</p>


##  Libraries used
</div>

- Keras  [1] -- a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
- Scikit-learn [2] is a free software machine learning library for Python.

##  REFERENCES

[1] F. Chollet, "Keras," 2015, https://keras.io

[2] F. Pedregosa, V. Gael, G. Alexandre, M. Vincent, T. Bertrand, G. Olivier, M. Blondel, P. Prettenhofer, W. Ron, D. Vincent, V. Jake, P. Alexandre, C. David and B. Matthieu, "Scikit-learn: Machine learning in Python.," Journal of Machine Learning Research, p. 2825–2830, 2011. 