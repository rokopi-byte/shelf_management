# Shelf Management

Dataset and Codebase for "Shelf Management: a Deep Learning-Based system for shelf visual monitoring".

[Paper](https://doi.org/10.1016/j.eswa.2024.124635)

![flow](https://github.com/rokopi-byte/shelf_management/assets/23717373/068908ee-70c6-41a2-8f08-d4c255e3339a)

This code is about the recognition part. 

For the detection part please refer to the original 
[work](https://github.com/eg4000/SKU110K_CVPR19) 
with [this](https://drive.google.com/file/d/1f9tRzJSqjuUQzXz8WjJC0V_WD-8y_6wy/view?usp=sharing) trained model. 

For the Shelf Row detection please refer the original [work](https://github.com/Hanqer/deep-hough-transform) 
with [this](https://drive.google.com/file/d/1P68u_GcaCO1D3fH9eFGBobBorKMba1bk/view?usp=drive_link) trained model.
Dataset con be found [here](https://figshare.com/articles/dataset/SHARD_-_SHelf_mAnagement_Row_Dataset/24100695)

Download the SHAPE dataset [here](https://figshare.com/articles/dataset/SHAPE_-_SHelf_mAnagement_Product_datasEt/24100704) and unzip in the main folder. _training_set_ and _test_set_ should contain directly
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

### Cite

```
@article{PIETRINI2024124635,
title = {Shelf Management: A deep learning-based system for shelf visual monitoring},
journal = {Expert Systems with Applications},
volume = {255},
pages = {124635},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.124635},
url = {https://www.sciencedirect.com/science/article/pii/S0957417424015021},
author = {Rocco Pietrini and Marina Paolanti and Adriano Mancini and Emanuele Frontoni and Primo Zingaretti},
keywords = {Shelf management, Retail, Shelf monitoring, SKU recognition, Planogram compliance, Planogram},
abstract = {Shelf monitoring plays a key role in optimizing retail shelf layout, enhancing the customer shopping experience and maximizing profit margins. The process of automating shelf audit involves the detection, localization and recognition of objects on store shelves, including diverse products with varying attributes in unconstrained environments. This facilitates the assessment of planogram compliance. Accurate product localization within shelves requires the identification of specific shelf rows. To address the current technological challenges, we introduce “Shelf Management”, a deep learning-based system that is carefully tailored to redesign shelf monitoring practices. Our system can navigate the complexities of shelf monitoring by using advanced deep learning techniques and object detection and recognition models. In addition, a complex semantic module enhances the accuracy of detecting and assigning products to their designated shelf rows and locations. In particular, we recognize the lack of finely annotated datasets at the SKU level. As a contribution to the field, we provide annotations for two novel datasets: SHARD (SHelf mAnagement Row Dataset) and SHAPE (SHelf mAnagement Product dataset). These datasets not only provide valuable resources, but also serve as benchmarks for further research in the field of retail. A complete pipeline is designed using a RetinaNet architecture for object detection with 0.752 mAP, followed by a Deep Hough transform to detect shelf rows as semantic lines with an F1 score of 97%, and a product recognition step using a MobileNetV3 architecture trained with triplet loss and used as a feature extractor together with FAISS for fast image retrieval with an accuracy of 93% on top-1 recognition. Localization is achieved using a deterministic approach based on product detection and shelf row detection. Source code and datasets are available at https://github.com/rokopi-byte/shelf_management.}
}
```
