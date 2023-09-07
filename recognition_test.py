import argparse
from pathlib import Path
import tensorflow as tf
from recognition_training import read_image
import time
import faiss
import pickle


class PredictionCallback(tf.keras.callbacks.Callback):
    start_time = time.time()

    def on_predict_begin(self, logs=None):
        self.start_time = time.time()
        print("Starting prediction ...")

    def on_predict_end(self, logs=None):
        end_time = time.time() - self.start_time
        print(f"Inference time per Image: {end_time / image_count} seconds")


def countTop(totalCount, EANListTest, label_list, top):
    '''
    Count function for the accuracy metrics
    :param totalCount: Total number of EANs in test set
    :param EANListTest: List of EANs in the test set
    :param label_list: Labels from the gallery
    :param top: Number of results to consider, X for top-X
    :return: Count of EANs in test set correctly matched
    '''
    countTop = 0
    for i in range(totalCount):
        if EANListTest[i] in [label_list[I[i][j]] for j in range(top)]:
            countTop += 1
    return countTop


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir_gallery', type=str, required=True)
parser.add_argument('--data_dir_test', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--target_shape', type=int, required=True)
parser.add_argument('--embedding_size', type=int, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--gallery_embeddings', type=str, required=False)
args = parser.parse_args()

data_dir_gallery = args.data_dir_gallery
data_dir_test = args.data_dir_test
batch_size = args.batch_size
embedding_size = args.embedding_size
target_shape = (args.target_shape, args.target_shape)
gallery_embeddings = args.gallery_embeddings
data_dir_gallery = Path(data_dir_gallery, '**', '**', '*.jpg')
data_dir_test = Path(data_dir_test, '**', '**', '*.jpg')
list_ds = tf.data.Dataset.list_files(str(data_dir_gallery), shuffle=False)

image_count = len(list_ds)
label_list = []
for i in list_ds:
    label_list = label_list + [int(Path(i.numpy().decode()).parent.name)]

label_list_tf = tf.data.Dataset.from_tensor_slices(label_list)
completeDataset = list_ds.map(lambda x: read_image(x, target_shape))

completeDataset = completeDataset.batch(batch_size, drop_remainder=False)
completeDataset = completeDataset.prefetch(8)

embeddingNet = tf.keras.models.load_model(args.model)
embeddingTotale = None

# Inference on gallery, or loading from pickle file
if gallery_embeddings:
    file_name = gallery_embeddings
    features = None
    if Path(file_name).exists():
        with open(file_name, 'rb') as pickle_file:
            embeddingTotale = pickle.load(pickle_file)
else:
    embeddingTotale = embeddingNet.predict(completeDataset, callbacks=[PredictionCallback()])

pickle.dump(embeddingTotale, open('embeddings.pickle', 'wb'))

list_all_images_for_embedding = (list(list_ds.as_numpy_iterator()))
numFotoTotale = len(list_all_images_for_embedding)
EANTotale = []
imageTotale = []
cat_totale = []
for i in list_all_images_for_embedding:
    EANTotale = EANTotale + [Path(i.decode('UTF-8')).parent.name]
    cat_totale = cat_totale + [Path(i.decode('UTF-8')).parent.parent.name]
    imageTotale = imageTotale + [Path(i.decode('UTF-8')).stem]

EAN_imageTotale = []
for i in range(numFotoTotale):
    EAN_imageTotale = EAN_imageTotale + [EANTotale[i] + "_" + imageTotale[i] + "_" + cat_totale[i]]

dictTotale = {key: [] for key in EAN_imageTotale}
for k, v in zip(EAN_imageTotale, embeddingTotale):
    dictTotale[k].append(v)

# Inference on test set
list_dsTest = tf.data.Dataset.list_files(str(data_dir_test), shuffle=False)
image_countTest = len(list_dsTest)
EANListTest = []
imageListTest = []
for i in list_dsTest:
    EANListTest = EANListTest + [int(Path(i.numpy().decode()).parent.name)]
    imageListTest = imageListTest + [Path(i.numpy().decode()).name]

datasetTest = list_dsTest.map(lambda x: read_image(x, target_shape))
datasetTest = datasetTest.batch(batch_size, drop_remainder=False)
datasetTest = datasetTest.prefetch(8)
embeddingTest = embeddingNet.predict(datasetTest, callbacks=[PredictionCallback()])

# Similarity Search
index = faiss.IndexFlatIP(embedding_size)
faiss.normalize_L2(embeddingTotale)
index.add(embeddingTotale)
k = 10
start_time = time.time()
D, I = index.search(embeddingTest, k)
print("%s seconds" % ((time.time() - start_time) / image_countTest))

totalCount = len(EANListTest)

# Accuracy top1, top5 and top10
print(f'Accuracy: {countTop(totalCount, EANListTest, label_list, 1) / totalCount}')
print(f'Accuracy top 5: {countTop(totalCount, EANListTest, label_list, 5) / totalCount}')
print(f'Accuracy top 10: {countTop(totalCount, EANListTest, label_list, 10) / totalCount}')
