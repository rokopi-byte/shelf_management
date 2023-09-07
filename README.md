# Shelf Management

Dataset and Codebase for "Shelf Management: a Deep Learning-Based system for shelf visual monitoring".

This code is about the recognition part. 

For the detection part please refer to the original 
[work](https://github.com/eg4000/SKU110K_CVPR19) 
with [this](https://drive.google.com/file/d/1f9tRzJSqjuUQzXz8WjJC0V_WD-8y_6wy/view?usp=sharing) trained model. 

For the Shelf Row detection please refer the original [work](https://github.com/Hanqer/deep-hough-transform) 
with [this](https://drive.google.com/file/d/1P68u_GcaCO1D3fH9eFGBobBorKMba1bk/view?usp=drive_link) trained model.
Dataset con be found [here](https://figshare.com/account/articles/24100695)

Download the SHAPE dataset [here](https://figshare.com/s/3cc44298812b0427aa05) and unzip in the main folder. _training_set_ and _test_set_ should contain directly
the category folders (numbered), inside each category folder there are the EAN folders (masked, so 1,2,3 again) which
represent the labels for the images in each folder.

Current code uses the best models according to the paper, other configurations for backbones and search strategies
can be tuned directly in the code.

Then install requirements using:

    pip install -r requirements.txt
If you have a GPU you can use faiss-gpu for the similarity search in GPU.

### Training:

Tensorboard callback is used to monitor the training process.

    python recognition_training.py --target_shape 224 --embedding_size 256 --batch_size 64 --num_epochs 30 
                                    --data_dir training_set

### Test

    python recognition_test.py --target_shape 224 --data_dir_gallery training_set --data_dir_test test_set 
                                --batch_size 64 --embedding_size 256 --model model

Optionally with argument _--gallery_embeddings_ is possible to provide a pickle file containing the gallery embeddings.
If not provided are computed on the fly and stored in a embeddings.pickle file locally.