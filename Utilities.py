import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" #to suppress some unnecessary warnings

import tensorflow as tf, keras, numpy as np, random


"""
    Data loaders
"""

# load an image from a file specified by 'path', returns numpy array
def load_img(path, img_size):
    image = keras.preprocessing.image.load_img(path)
    image_arr = keras.preprocessing.image.img_to_array(image)
    image_arr = tf.image.resize(image_arr, img_size)
    return image_arr / 255


# DataLoader for Triplet loss, used to train a Siamese Network
class DataLoaderTriplet(tf.keras.utils.Sequence):

    def __init__(self, dataset_root_path, batch_size, img_size=(250, 250), *args, **kwargs):
        super().__init__(**kwargs)
        
        self.img_size = img_size
        self.batch_size = batch_size

        self.dataset_root_path = dataset_root_path
        self.classes_paths = []
        
        # scan all classes directories of the dataset
        for class_dir in os.scandir(dataset_root_path):
            files = os.listdir(class_dir.path)
            self.classes_paths.append([])

            for file in files:
                self.classes_paths[-1].append(os.path.join(class_dir.path, file))
        
        # shuffle the paths of the images of the dataset
        np.random.shuffle(self.classes_paths)

        self.batches_num = np.floor(len(self.classes_paths) / (self.batch_size * 2)).astype(np.int32)
    
    # the training algorithm (model.fit()) will call this function to get the (n)th batch of the dataset
    def __getitem__(self, n):

        # allocate memory for the batch
        X1 = np.zeros((self.batch_size, self.img_size[1], self.img_size[0], 3))
        X2 = np.zeros((self.batch_size, self.img_size[1], self.img_size[0], 3))
        X3 = np.zeros((self.batch_size, self.img_size[1], self.img_size[0], 3))

        # get two sub-batches
        i = n * self.batch_size
        sub_batch_1 = self.classes_paths[i : i + (self.batch_size)]
        sub_batch_2 = self.classes_paths[i + (self.batch_size) : i + (self.batch_size * 2)]

        # make sub-batch of similar pairs
        for i in range(self.batch_size):
            person_1 = sub_batch_1[i]
            person_2 = sub_batch_2[i]

            person_1_images = random.sample(person_1, 2)

            anchor = person_1_images[0]
            positive = person_1_images[1]
            negative = random.sample(person_2, 1)[0]
            
            X1[i] = load_img(positive, self.img_size)
            X2[i] = load_img(anchor, self.img_size)
            X3[i] = load_img(negative, self.img_size)
        
        # since the model has three inputs, X1(postive), X2(anchor) and X3(negative) has to be grouped into a tuple.
        return (X1, X2, X3)

    # returns the number of batches in the dataset
    def __len__(self):
        return self.batches_num
    
    # called at the end of the epoch
    def on_epoch_end(self):
        # shuffle the paths of the images of the dataset
        np.random.shuffle(self.classes_paths)


# DataLoader for Contrastive loss, used to train a Siamese Network
class DataLoaderContrastive(tf.keras.utils.Sequence):
    
    # the argument (positive_ratio): positive to negative ratio of pairs for each batch, must be in range [0.0, 1.0]
    def __init__(self, dataset_root_path, batch_size, positive_ratio, img_size=(250, 250), *args, **kwargs):
        super().__init__(**kwargs)
        
        self.img_size = img_size
        self.batch_size = batch_size

        self.dataset_root_path = dataset_root_path
        self.classes_paths = []
        
        # scan all classes directories of the dataset
        for class_dir in os.scandir(dataset_root_path):
            files = os.listdir(class_dir.path)
            self.classes_paths.append([])

            for file in files:
                self.classes_paths[-1].append(os.path.join(class_dir.path, file))
        
        # shuffle the paths of the images of the dataset
        np.random.shuffle(self.classes_paths)

        '''
            calculate the total number of batches that would cover the whole dataset in a single epoch,
            The dataset will be divided into groups of three sub-batches,
            - One sub-batch is used for similar pairs, each class images is paired with another image from the same class,
            - The other two sub-batches are used to pair with each other to make dissimilar pairs
            The final result is the concatenation of the two sub-batches, with size of (batch_size)
        '''
        self.num_positive_pairs = np.round(self.batch_size * positive_ratio).astype(np.int32)
        self.num_negative_pairs = self.batch_size - self.num_positive_pairs
        self.sub_batches_size = self.num_positive_pairs + self.num_negative_pairs * 2

        self.batches_num = np.floor(len(self.classes_paths) / self.sub_batches_size).astype(np.int32)
    
    # the training algorithm (model.fit()) will call this function to get the (n)th batch of the dataset
    def __getitem__(self, n):

        # allocate memory for the batch
        X1 = np.zeros((self.batch_size, self.img_size[1], self.img_size[0], 3))
        X2 = np.zeros((self.batch_size, self.img_size[1], self.img_size[0], 3))
        Y = np.zeros((self.batch_size, 1))

        # get three sub-batches
        i = n * self.sub_batches_size
        num_p = self.num_positive_pairs
        num_n = self.num_negative_pairs

        sub_batch_1 = self.classes_paths[i : i + num_p]
        sub_batch_2 = self.classes_paths[i + num_p : i + num_p + num_n]
        sub_batch_3 = self.classes_paths[i + num_p + num_n : i + num_p + num_n * 2]

        # make sub-batch of similar pairs
        for i in range(num_p):
            person_1 = sub_batch_1[i]

            person_1_images = random.sample(person_1, 2)
            X1[i] = load_img(person_1_images[0], self.img_size)
            X2[i] = load_img(person_1_images[1], self.img_size)
            Y[i] = 1

        # make another sub-batch of dissimilar pairs
        for i in range(num_n):
            person_2 = sub_batch_2[i]
            person_3 = sub_batch_3[i]

            X1[i + num_p] = load_img(random.sample(person_2, 1)[0], self.img_size)
            X2[i + num_p] = load_img(random.sample(person_3, 1)[0], self.img_size)
            Y[i + num_p] = 0
        
        # Since the model has two inputs, X1 and X2 has to be grouped into a tuple.
        return (X1, X2), Y

    # returns the number of batches in the dataset
    def __len__(self):
        return self.batches_num
    
    # called at the end of the epoch
    def on_epoch_end(self):
        # shuffle the paths of the images of the dataset
        np.random.shuffle(self.classes_paths)


def get_dataset_contrastive_with_prefetching(dataset_root_path, batch_size = 32, positive_ratio=0.2, image_size = (250, 250)):
    '''
        This is a simple function that turns a data loader into a tensorflow dataset with prefetching enabled 
        for better performance when training.
    '''
    h, w = image_size

    return tf.data.Dataset.from_generator(
        DataLoaderContrastive, 
        args = [dataset_root_path, batch_size, positive_ratio, (h, w)], 
        output_signature = ((tf.TensorSpec(shape = (batch_size, h, w, 3), dtype = tf.float32), tf.TensorSpec(shape = (batch_size, h, w, 3), dtype = tf.float32)), tf.TensorSpec(shape = (batch_size, 1), dtype = tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

def get_dataset_triplet_with_prefetching(dataset_root_path, batch_size = 32, image_size = (250, 250)):
    '''
        This is a simple function that turns a data loader into a tensorflow dataset with prefetching enabled 
        for better performance when training.
    '''
    h, w = image_size

    return tf.data.Dataset.from_generator(
        DataLoaderTriplet, 
        args = [dataset_root_path, batch_size, (h, w)], 
        output_signature = ((tf.TensorSpec(shape = (batch_size, h, w, 3), dtype = tf.float32), tf.TensorSpec(shape = (batch_size, h, w, 3), dtype = tf.float32), tf.TensorSpec(shape = (batch_size, h, w, 3), dtype = tf.float32)))
    ).prefetch(tf.data.AUTOTUNE)



"""
    Models
"""

class L2Normalization(keras.Layer):
    """
    Custom L2Normalization layer.
    This is used to normalize the embeddings obtained from 
    the embedding model.
    Ensures the output embeddings lie on a unit hypersphere.
    """
    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)

    def call(self, inputs):
        # Perform L2 normalization along the last axis
        return tf.math.l2_normalize(inputs, axis=-1)


@keras.saving.register_keras_serializable("Patches")
class Patches(keras.layers.Layer):
    """
        This is the patching layer.
        It takes as input an image or a feature map, and it
        return as output the same image but cut into patches
        of size patch_size. 

        patch_size: The size length of each patch (assume square patches)
    """
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size
    
    def call(self, images):
        input_shape = keras.ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size = self.patch_size)
        patches = keras.ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels
            )
        )

        return patches
    

@keras.saving.register_keras_serializable("PatchEncoder")
class PatchEncoder(keras.layers.Layer):
    """
        This is the patch encoding layers.
        It takes as input the patches output by the Patches layer, 
        and it projects the patches into the required projection_dim and adds
        positional information to the encoded patches.

        num_patches: The number of patches each image is split into.

        projection_dim: The required projection size of each patch.
    """
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = keras.layers.Dense(units = self.projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim = num_patches,
            output_dim = projection_dim
        )
    
    def call(self, patches):
        positions = keras.ops.expand_dims(
            keras.ops.arange(start = 0, stop = self.num_patches, step = 1), axis = 0
        )
        projected_patches = self.projection(patches)
        encoded = projected_patches + self.position_embedding(positions)
        
        return encoded

    def get_config(self):
        config = super().get_config()
        return {**config, 'num_patches' : self.num_patches, 'projection_dim' : self.projection_dim}


def mlp(x, hidden_units, dropout_rate, projection_dim):
    """
    This function implements a multilayer perceptron that will be used 
    inside the transformer after each multihead attention layer.

    hidden_units: The number of neurons in each layer.

    dropout_rate: The dropout rate to be used between each hidden unit.
    
    projection_dim: The number of neurons in the output of the mlp.
    """

    for units in hidden_units + [projection_dim]:
        x = keras.layers.Dense(units, activation = keras.activations.gelu)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


def get_embedding_model(projection_dim = 192, transformer_layers = 8, num_heads = 10, transformer_units = [256], embedding_size = 128, patch_size = 1, cnn_feature_map_layer = 'conv5_block3_out'):
    """
        This function returns the main model that will be used
        to calculate the embeddings of faces.

        The model used here relies on resnet50 for the cnn part
        and a custom transformer encoder for the ViT part.

        It accepts images of size image_size and it returns an 
        embedding feature vector of size embedding_size.

        projection_dim: The size of the vectors inside the transformer; 
                        each sublayer in the transformer (multihead attention and
                        mlp) should have an input and output of this size.
        
        transformer_layers: The number of layers in the transformer encoder.

        num_heads: The number of heads to be used in the multihead attention.

        transformer_units: The number of neurons inside the transformer for the
                            mlp sublayer.

        embedding_size: The final output size of the model.  

        patch_size: The number of pixels in each patch.

        cnn_feature_map_layer: The resnet50 layer that will be used to extract
                               the feature map from and feed it into the
                               transformer.                 
    """

    #CNN part
    image_size = 224
    
    #get the cnn
    resnet50 = keras.applications.ResNet50(
        include_top = False, weights = 'imagenet', 
        input_shape = (image_size, image_size, 3)
    )
    

    #get the output layer of the cnn that will be fed into the transformer
    resnet50_output_layer = resnet50.get_layer(cnn_feature_map_layer)
    resnet50_output = resnet50_output_layer.output
    
    #cut the cnn
    cnn = keras.models.Model(inputs = resnet50.inputs, outputs = resnet50_output)

    #transformer part

    #get the size of the feature map output of resnet50, which will be the same
    #size of the input feature map into the transformer
    resnet50_output_feature_map_size = resnet50_output_layer.output.shape[1]
    

    # calculate the total number of patches that will enter into the transformer
    num_patches = (resnet50_output_feature_map_size//patch_size)**2


    #build the transformer
    inputs = resnet50.input

    #pass the input through the cnn
    cnn_output_feature_map = cnn(inputs)
    
    #patch the feature maps
    patches = Patches(patch_size)(cnn_output_feature_map)

    #encode the patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)


    #Create multiple transformer encoder layers
    for _ in range(transformer_layers):
        #first layer normalization (normalize the input to the encoder)
        x1 = keras.layers.LayerNormalization(epsilon = 1e-9)(encoded_patches)

        #Multihead attention layer
        attention_output = keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = projection_dim,
            dropout = 0.1,
        )(x1, x1)

        #skip connection 1
        x2 = keras.layers.Add()([attention_output, encoded_patches])

        #second layer normalization
        x3 = keras.layers.LayerNormalization(epsilon = 1e-9)(x2)

        #MLP
        x3 = mlp(x3, hidden_units = transformer_units, dropout_rate = 0.1, projection_dim = projection_dim)

        #skip connection 2
        encoded_patches = keras.layers.Add()([x3, x2])
    
    #Get only the first element in the sequence for all batches and all neurons 
    transformer_output = encoded_patches[:, 0, :]
    transformer_output = keras.layers.Dropout(0.5)(transformer_output)
    
    #build the final mlp
    outputs = keras.layers.Dense(1024, activation = 'relu')(transformer_output)
    outputs = keras.layers.Dropout(0.5)(outputs)
    
    #set the output activation to linear since we will use either
    #contrastive loss or triplet loss, and the distance metric is
    #the euclidean distance
    outputs = keras.layers.Dense(embedding_size, activation = 'linear')(outputs)

    #normalize the embedding so that it lies on a unit hypersphere
    outputs = L2Normalization()(outputs)
    
    return keras.models.Model(inputs = inputs, outputs = outputs)