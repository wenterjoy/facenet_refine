Facenet  
This is a TensorFlow implementation of the face recognizer described in the paper "FaceNet: A Unified Embedding for Face Recognition and Clustering". The project also uses ideas from the paper "Deep Face Recognition" from the Visual Geometry Group at Oxford.
The system directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as feature vectors. 
The method uses a deep convolutional network trained to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning approaches. To train, the authors use triplets of roughly aligned matching / non-matching face patches generated using a novel online triplet mining method. The benefit of our approach is much greater representational efficiency: we achieve state-of-the-art face recognition performance using only 128-bytes per face. 
On the widely used Labeled Faces in the Wild (LFW) dataset, our system achieves a new record accuracy of 99.63%. On YouTube Faces DB it achieves 95.12%. Our system cuts the error rate in comparison to the best published result by 30% on both datasets. 
We also introduce the concept of harmonic embeddings, and a harmonic triplet loss, which describe different versions of face embeddings (produced by different networks) that are compatible to each other and allow for direct comparison between each other. 
Original github: https://github.com/davidsandberg/facenet
Framework: Tensorflow + Python 3.5/3.6
Step 1. Download code from https://github.com/wenterjoy/facenet_refine.git
'''git clone https://github.com/wenterjoy/facenet_refine.git
Step 2. Access the folder: facenet_refine , then install all the packages according to requirement.txt
pip install -r requirements.txt
Step 3. Download the pre-trained model
https://ibm.box.com/s/91phz1tmawjkg8n99jubqffrvbnewwfm
 Extract the file to the specified directory, suggested directory is < facenet_refine/data/>

Step 4: Image preprocess - running face alignment program 

Access to folder: facenet_refine > src > align 
python  align_dataset_mtcnn.py  \
 < directory with unaligned images >  \  
< directory with aligned face thumbnails >
