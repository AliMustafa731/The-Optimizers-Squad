import keras
import tensorflow as tf
from keras import ops


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


def get_backbone_model(projection_dim = 128, transformer_layers = 8, num_heads = 8, transformer_units = [512], patch_size = 4, cnn_feature_map_layer = 'conv3_block4_out', image_size = 128):
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
    
    #get the cnn
    resnet50 = keras.applications.ResNet50(
        include_top = False, 
        input_shape = (image_size, image_size, 3),
    )
    
    
    #get the output layer of the cnn that will be fed into the transformer
    resnet50_output_layer = resnet50.get_layer(cnn_feature_map_layer)
    resnet50_output = resnet50_output_layer.output
    
    #cut the cnn
    cnn = keras.models.Model(inputs = resnet50.input, outputs = resnet50_output)

    #transformer part

    #get the size of the feature map output of resnet50, which will be the same
    #size of the input feature map into the transformer
    resnet50_output_feature_map_size = resnet50_output_layer.output.shape[1]
    

    # calculate the total number of patches that will enter into the transformer
    num_patches = (resnet50_output_feature_map_size//patch_size)**2


    #build the transformer
    inputs = cnn.input

    #Add rescaling
    rescaling = keras.layers.Rescaling(scale = 1.0/255)(inputs)

    #pass the input through the cnn
    cnn_output_feature_map = cnn(rescaling)
    
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
    

    transformer_output = keras.layers.Flatten()(encoded_patches)
    #build the final mlp
    transformer_output = keras.layers.Dropout(0.25)(transformer_output)
    outputs = keras.layers.Dense(512, activation = 'relu')(transformer_output)

    return keras.models.Model(inputs = inputs, outputs = outputs, name = 'backbone_model')


def get_embedding_model(backbone_model_weights_path = None, embedding_size = 128):
    backbone_model = get_backbone_model()
    if backbone_model_weights_path != None:
        backbone_model.load_weights(backbone_model_weights_path)

    inputs = backbone_model.input
    features = backbone_model(inputs)
    embedding_output = keras.layers.Dense(embedding_size, activation = 'linear')(features)
    normalized = L2Normalization()(embedding_output)
    embedding_model = keras.models.Model(inputs = inputs, outputs = normalized)
    return embedding_model


def euclidean_distance(vectors):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vectors: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vectors
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, keras.backend.epsilon()))


def contrastive_loss(margin = 1):

    def c_loss(y_true, y_pred):
        """Calculates the contrastive loss.

            Arguments:
                y_true: List of labels, each label is of type float32.
                y_pred: List of predictions of same length as of y_true,
                        each label is of type float32.

            Returns:
                A tensor containing contrastive loss as floating point value.
        """
        y_true = tf.cast(y_true, y_pred.dtype) #cast them to the same type to avoid errors

        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return c_loss



def get_siamese_model(backbone_weights_path = None, image_size = 128, with_augmentation = False):
    if backbone_weights_path != None:
        embedding_model = get_embedding_model(backbone_weights_path)
    else:
        embedding_model = get_embedding_model()

    input1 = keras.layers.Input(shape = (image_size, image_size, 3))
    input2 = keras.layers.Input(shape = (image_size, image_size, 3))

    if with_augmentation:
        augmentor = keras.models.Sequential([
            keras.layers.RandomFlip(),
            keras.layers.RandomBrightness(factor = 0.1),
            keras.layers.RandomContrast(factor = 0.1),
            keras.layers.RandomRotation(factor = 0.02),
            keras.layers.RandomZoom(height_factor = 0.05, width_factor = 0.05)
        ], name = 'augmentor')

        augmented1 = augmentor(input1)
        augmented2 = augmentor(input2)
        embedding_1 = embedding_model(augmented1)
        embedding_2 = embedding_model(augmented2)
    else:
        embedding_1 = embedding_model(input1)
        embedding_2 = embedding_model(input2)

    distance = keras.layers.Lambda(euclidean_distance, output_shape = (1,))(
        [embedding_1, embedding_2]
    )
    
    siamese = keras.models.Model(inputs = [input1, input2], outputs = distance)
    return siamese


def get_compiled_siamese_model(weights_path = None):
    siamese_model = get_siamese_model(with_augmentation = False)

    siamese_model.compile(
        loss = contrastive_loss(),
        optimizer = keras.optimizers.Adam(learning_rate = 0.0001, weight_decay = 0.0001),
        metrics = ['accuracy']
    )

    if weights_path != None:
        siamese_model.load_weights(filepath = weights_path)

    return siamese_model
