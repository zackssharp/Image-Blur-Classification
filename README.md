
# CERTH ImageBlurDataset Classification
by [Dhruv Patel]([Github](https://github.com/dhruvbpatel/))

**Problem Statement:** 
	Image quality detection has always been an arduous problem to solve in computer vision.In this project, We will  come up with features and models to build a classifier which will predict whether a given image is blurred.

DataSet Link: [CERTH_ImageBlurDataset](http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip)
-

This repository contains code for Blurred Image classification.

**Dependecies:**
All the dependencies necessary for running the code is mentioned in **requirements.txt** file.

 - Firstly download the zip file and extract it

you can simple install all of them by running:

    pip install -r requirements.txt
   
   ---
**File Description**
| No. | FileName | Description  |
|--|--|--|
|1| CNN (ResNet 50) Blur_Image_Classification | Code for CNN model training and testing. Got an aggregated accuracy of ~85% using ResNet 50 Architecture.|
| 2 | Laplacian Detection on DigitalBlurSet (96% accuracy).py | Code for Laplacian Technique on Evaluation set which got 96% accuracy on DigitalBlur Set |
|3| Laplacian Detection on NaturalBlurSet (76% accuracy)| Code for Laplacian Technique on Evaluation Set which got 76% accuracy on NaturalBlur Set | 
|4|model_1_stage_3 | Saved Model after CNN training
|5| preds.csv| Prediction results of CNN model after testing on evaluation set |
|6| cnn_blur_classification.py | fastai v2 port of the CNN notebook. Trains from scratch and scores both evaluation sets (see **Reproduced Results**) |
|7| finetune_chili.py | Fine-tunes the CERTH CNN on the chili dataset and compares it against the Laplacian baseline (see **Fine-tuning on the chili dataset**) |


All code files are available in both Jupyter notebook(.IPYNB) as well as python (.py) files.

 - This code was developed using **Google Colab GPU** runtime. So it is Highly advisible to execute this code using the same inorder to work hasslelessly.

---
**Methods Used:**

 - **CNN Based Image Classification using ResNet50 architecture**
	 -	 Using the training dataset the CNN model was trained. It incorporated the use of FASTAI library for developing the model , training and testing it.
	 -	using fastai I was able to train a CNN model using Resnet50 architecture.
	 -	The trained model was able to achieve an accuracy of ~85% on the dataset
	 -	The steps of buidling CNN involved preprocessing of Images and creating model , training it on dataset, Hyperparameter tuning , testing on validation set and then finally making prediction.
	 
 - **Laplacian Based Technique for Classifying if Images is Blurred or not**
			-	 Using Laplace transformation and calculating the variance in frequency of images we can build a classifier which can classify if an image if blurred or not.
			-	Using OpenCV and Numpy library we can vectorize the images and apply laplace method to get the variance
			-	also we can tune hyperparameter to put a threshold for classification.
			-	We were able to achieve and accuracy of ~96% on DigitalBlurSet and ~76% on NaturalBlurSet


**Feature Engineering and Inference Outcomes.**
	 -  The images in the Dataset were already categorized but the problem was that they were of different sizes
	 - Inorder to train our model effectively we reformated them into a same size and loaded into ImageDataBunch
	 - Renaming Images was also an important decision so that we can remove all mismatch in naming convention
	 - Also required preprocessing were performed before training to get most out of trained model
	 - After training upon testing in inference of trained model we found that many images were misclassified and we also displayed them
	 - Almost ~10-15% image predictions were slightly misclassified
	 - After Tuning hyperparameters and again training model we found the best hyperparameter as => 1e-03 as learning rate
	 - Further improvement of the model can be done with more images in dataset.

---
**Reproduced Results**

`model_1_stage_3` and `preds.csv` are not in the repository and the notebook is written for fastai v1, so the CNN was ported to fastai v2 (`cnn_blur_classification.py`) and retrained. Run everything from the repository root with the dataset extracted to `./CERTH_ImageBlurDataset`:

    python cnn_blur_classification.py                  # ResNet-34, what the notebook code trains
    python cnn_blur_classification.py --arch resnet50
    MPLBACKEND=Agg python "Laplacian Detection on DigitalBlurSet (96% accuracy).py"
    MPLBACKEND=Agg python "Laplacian Detection on NaturalBlurSet (76% accuracy).py"

The CNN script follows the notebook's recipe (80/20 split, 224px, 4 frozen epochs then 2 unfrozen) and saves its weights to `CERTH_ImageBlurDataset/TrainingSet/models/` and its predictions to `preds_<arch>.csv`. The numbers below were obtained with fastai 2.8.12, torch 2.14.1, pandas 2.3.3 and scikit-learn 1.7.2 (`openpyxl` is also needed to read the `.xlsx` files), not the versions pinned in `requirements.txt`.

| | ResNet-34 | ResNet-50 | Laplacian |
|--|--|--|--|
| Validation accuracy, 3 classes (200 held-out training images) | 86.0% | 84.0% | - |
| Validation accuracy, blurred vs undistorted | 87.0% | 85.5% | - |
| DigitalBlurSet accuracy (480 images) | 83.1% | 82.3% | 96.25% |
| NaturalBlurSet accuracy (1000 images) | 76.8% | 73.7% | 76.5% |

 - The CNN numbers are from a single run with seed 42 and are new weights, not the original `model_1_stage_3`.
 - The evaluation set scores for the CNN treat either blurred class as "blurred" and compare against the spreadsheet labels.
 - On the evaluation sets the CNN is no better than the Laplacian method on natural blur and well behind it on digital blur.

---
**Fine-tuning on the chili dataset**

`finetune_chili.py` fine-tunes the CERTH CNN on a separate (not included) dataset of 2,070 camera frames laid out as `<data>/{good,bad}/{fwd,back,overhead}/<id>_<clip>_frameNN.jpg`, where `bad` means blurred (about 10% of the frames). Run `cnn_blur_classification.py` first, since it provides the starting weights:

    python finetune_chili.py [--arch resnet34|resnet50] [--size 448] [--init certh|imagenet] [--data ~/p1/data/chili]

The split is 60/20/20 train/valid/test, grouped by event id so that frames from one event never land in both training and test. The loss is class weighted. Each run saves its model to `models/chili_<arch>_<init>_<size>.pth` and writes `chili_results_*.csv` and `chili_preds_*.csv`.

Results on the test split (395 frames, 37 `bad`), with `bad` as the positive class and a threshold of 0.5:

| Model | Accuracy | Precision | Recall | F1 |
|--|--|--|--|--|
| Always "good" | 90.6% | - | 0% | - |
| Laplacian variance (threshold tuned on train+valid) | 98.2% | 100% | 81.1% | 0.896 |
| CERTH ResNet-34, no fine-tuning, 448px | 70.4% | 24.0% | 100% | 0.387 |
| Fine-tuned ResNet-34, 224px | 95.9% | 83.9% | 70.3% | 0.765 |
| Fine-tuned ResNet-34, 448px (default) | 98.0% | 83.7% | 97.3% | 0.900 |
| Fine-tuned ResNet-34, 672px | 97.5% | 80.0% | 97.3% | 0.878 |
| Fine-tuned ResNet-34, 448px, ImageNet init | 97.7% | 81.8% | 97.3% | 0.889 |
| Fine-tuned ResNet-50, 448px | 98.2% | 87.5% | 94.6% | 0.909 |

 - Resolution matters most: at 224px the fine-tuned CNN is worse than the Laplacian baseline, so 448px is the default.
 - The Laplacian baseline has no false alarms but misses 7 of the 37 blurred frames, where the 448px ResNet-34 misses 1.
 - The 448px and larger variants are within noise of each other (one test image is 0.25% accuracy), and starting from the CERTH weights shows no clear benefit over plain ImageNet weights.
 - The `good`/`bad` labels are the predictions of a vision language model (gemma4) and were not checked by hand, so these numbers measure agreement with that model.
