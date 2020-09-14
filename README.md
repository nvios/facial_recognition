# Live Facial Recognition

Live Facial Recogition of person using web-cam. The framework can detect multiple people in a single frame. The model was developed as part of a team project work to explore the integration of different ML and deep learning tools in a single application. 

# Model

1. Every i’th frame in the video is passed to the MTCNN module which detects the faces and provides the bounding box.
2. The aligned image is then passed to facenet model.
3. The embedding is the passed to a classifier.

# Face Detection (MTCNN)

Multi-task Cascaded Convolutional Networks is a three stage cascade deep neural network that predicts face and its location (bounding box).  This framework exploits the inherent correlation between detection and alignment.

**1 (Proposal Network -PNET)**: A Fully convolutional network, called Proposal Network (P-Net) is used to obtain the facial windows and their bounding box regression vectors. Facial candidates are further calibrated using bounding boxes followed by a non maximal suppression.

**2: (Refine Network-RNET)**: All candidates are fed to a Refine Network (R-Net), which further rejects a large number of false candidates, performs calibration with bounding box regression followed by non maximal suppression

**3: (Output Network- ONET)**: Similar to the second stage, but in this stage face regions are identified with more supervision. In particular, the network will output five facial landmarks positions.

# Facenet (Feature Extraction)

FaceNet, learns a mapping from face images to a compact 128-D Euclidean space Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as feature vectors. 
The pre-trained model of Inception Resnet v1 trained on CASIA-Webface dataset is being used and it gives an accuracy of 98.72% 

# Classification

The extracted features from facenet are trained on RF and SVM classifier for the created dataset.


________________________________________________________________________________________________________ _ _ _


# Preparing the Dataset

For preparing the dataset, separate folders for each person need to be created. For each folder should have nearly 15-25 images of the person in different positions, keeping the camera in front of it.

# Running the Code

    python face_recog_live.py

**For Training the Classifier with the dataset**  
  
python train_classifier.py --input-dir 'INPUT IMAGE DIRECTORY' --model-path models/20170511-185253.pb --classifier-path 'PATH TO SAVE THE TRAINED MODEL' --num-threads 10 --num-epochs 5 --min-num-images-per-class 1

## Click <a href="https://nvios.github.io/luca_bontempi/"><strong>here</strong></a> to see my latest projects!
### Latest Deep Learning Project: <a href="https://luca-bontempi-object-detection.web.app/"><strong>Live Object Detection</strong></a>
