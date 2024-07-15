#%%
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

#%%
class SpatioExpert(layers.Layer):
    def __init__(self, num_classes=6, filters = 64, dense_units=32):
        super(SpatioExpert, self).__init__()
        self.conv1 = layers.Conv2D(filters=int(filters//4), kernel_size=3, strides=2, 
                                   padding='same')
        self.bn1 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act1 = layers.Activation('swish')
        self.conv2 = layers.DepthwiseConv2D(kernel_size=3, strides=2, depth_multiplier=1, padding='same', 
                                            use_bias=False)
        self.bn2 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act2 = layers.Activation('swish')

        self.conv3 = layers.Conv2D(filters=int(filters//2), kernel_size=3, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act3 = layers.Activation('swish')
        self.conv4 = layers.DepthwiseConv2D(kernel_size=3, strides=2, depth_multiplier=1, padding='same', 
                                            use_bias=False)
        self.bn4 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act4 = layers.Activation('swish')

        self.conv5 = layers.Conv2D(filters=filters, kernel_size=3, strides=2, padding='same')
        self.bn5 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act5 = layers.Activation('swish')
        self.conv6 = layers.DepthwiseConv2D(kernel_size=3, strides=2, depth_multiplier=1, padding='same', 
                                            use_bias=False)
        self.bn6 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act6 = layers.Activation('swish')

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(dense_units)
        self.fc2 = layers.Dense(num_classes)
        self.drop = layers.Dropout(0.2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x) 
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act6(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.drop(x)
    
class SpatioExpertMLP(layers.Layer):
    def __init__(self, num_classes=6, dense_units=32):
        super(SpatioExpertMLP, self).__init__()
        self.fc1 = layers.Dense(dense_units)
        self.fc2 = layers.Dense(int(dense_units*2))
        self.fc3 = layers.Dense(num_classes)
        self.drop = layers.Dropout(0.2)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.drop(x)
    
class TemporalExpert(layers.Layer):
    def __init__(self, num_classes, input_shape, filters=64):
        super(TemporalExpert, self).__init__()
        self.max_pool = layers.MaxPooling2D(pool_size=(input_shape[-3], 1))
        self.avg_pool = layers.AveragePooling2D(pool_size=(input_shape[-3], 1))
        self.reshape1 = layers.Reshape((118, 1))
        self.reshape2 = layers.Reshape((118, 1))
        self.add = layers.Add()

        self.conv1 = layers.Conv1D(filters=int(filters//4), kernel_size=3, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act1 = layers.Activation('swish')
        self.conv2 = layers.DepthwiseConv1D(kernel_size=3, strides=2, padding='same')
        self.bn2 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act2 = layers.Activation('swish')

        self.conv3 = layers.Conv1D(filters=int(filters//2), kernel_size=3, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act3 = layers.Activation('swish')
        self.conv4 = layers.DepthwiseConv1D(kernel_size=3, strides=2, padding='same')
        self.bn4 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act4 = layers.Activation('swish')

        self.conv5 = layers.Conv1D(filters=filters, kernel_size=3, strides=2, padding='same')
        self.bn5 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act5 = layers.Activation('swish')
        self.conv6 = layers.DepthwiseConv1D(kernel_size=3, strides=2, padding='same')
        self.bn6 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
        self.act6 = layers.Activation('swish')

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='swish')
        self.dropout = layers.Dropout(0.25)
        self.dense2 = layers.Dense(num_classes)

    def call(self, inputs):
        max_pool = self.max_pool(inputs)
        avg_pool = self.avg_pool(inputs)
        max_pool = self.reshape1(max_pool)
        avg_pool = self.reshape2(avg_pool)
        x = self.add([max_pool, avg_pool])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act6(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)
    
class Gating(layers.Layer):
    def __init__(self, gate_units=64, num_experts=6):
        super(Gating, self).__init__()
        # self.flatten = layers.Flatten()
        self.dense = layers.Dense(gate_units, activation='relu')
        self.expert_weights = layers.Dense(num_experts, activation='softmax')
        
    def call(self, x):
        # x = self.flatten(x)
        x = self.dense(x)
        return self.expert_weights(x)
    
class Gating2(layers.Layer):
    def __init__(self, input_shape, gate_units=32, num_experts=6):
        super(Gating2, self).__init__()
        self.max_pool = layers.MaxPooling2D(pool_size=(input_shape[-3], 1))
        self.avg_pool = layers.AveragePooling2D(pool_size=(input_shape[-3], 1))
        self.add = layers.Add()
        
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(gate_units, activation='swish')
        self.expert_weights = layers.Dense(num_experts, activation='softmax')
        self.drop = layers.Dropout(0.2)
        
    def call(self, inputs):
        max_pool = self.max_pool(inputs)
        avg_pool = self.avg_pool(inputs)
        x = self.add([max_pool, avg_pool])
        
        x = self.flatten(x)
        x = self.dense(x)
        x = self.expert_weights(x)
        return self.drop(x)
    
class STMoE(layers.Layer):
    def __init__(self, input_shape, spa_filters=64, temp_filters=64, 
                 temp_input_shape=(128, 118, 1),
                 gate_units=32, num_classes=6, 
                 num_spatio_experts=5, num_temp_experts=5, 
                 epsilon=1e-8, top_k=5):
        super(STMoE, self).__init__()
        self.spatio_experts = [SpatioExpert(num_classes, spa_filters) for _ in range(num_spatio_experts)]
        self.temp_experts = [TemporalExpert(num_classes, temp_input_shape, temp_filters) for _ in range(num_temp_experts)]
        self.gating = Gating2(input_shape, gate_units, (num_spatio_experts + num_temp_experts))
        self.softmax = layers.Activation('softmax')
        self.epsilon = epsilon
        self.top_k = top_k

    def call(self, x):
        weights = self.gating(x)
        random_noise = self.epsilon * tf.random.normal((1,))
        weights += random_noise
        
        top_k_values, top_k_indices = tf.math.top_k(weights, k=self.top_k)

        spa_expert_outputs = [expert(x) for expert in self.spatio_experts]
        temp_expert_outputs = [expert(x) for expert in self.temp_experts]
        expert_outputs = spa_expert_outputs + temp_expert_outputs

        expert_outputs = tf.stack(expert_outputs, axis=1)  

        batch_size = tf.shape(x)[0]
        top_k_indices_expanded = tf.reshape(top_k_indices, (batch_size, self.top_k))
        top_k_expert_outputs = tf.gather(expert_outputs, top_k_indices_expanded, batch_dims=1, axis=1)

        # Broadcast the top k weights
        top_k_weights_expanded = tf.reshape(top_k_values, (batch_size, self.top_k, 1))
        top_k_weights_broadcasted = tf.broadcast_to(top_k_weights_expanded, tf.shape(top_k_expert_outputs))


        final_output = tf.reduce_sum(top_k_expert_outputs * top_k_weights_broadcasted, axis=1) + self.epsilon
        return self.softmax(final_output)

class MoE2(layers.Layer):
    def __init__(self, input_shape, spa_filters=64, 
                 gate_units=32, num_classes=6, 
                 num_experts=5, 
                 epsilon=1e-8, top_k=5, dropout_rate=0):
        super(MoE2, self).__init__()
        self.spatio_experts = [SpatioExpert(num_classes, spa_filters) for _ in range(num_experts)]
        self.gating = Gating2(input_shape, gate_units, num_experts)
        self.softmax = layers.Activation('softmax')
        self.epsilon = epsilon
        self.top_k = top_k
        self.dropout_rate = dropout_rate


    def call(self, x):
        weights = self.gating(x)
        
        dropout_mask = tf.nn.dropout(tf.ones_like(weights), rate=self.dropout_rate)
        weights *= dropout_mask

        # Add a small random value to the gating weights
        random_noise = self.epsilon * tf.random.normal((1,))
        weights += random_noise

        top_k_values, top_k_indices = tf.math.top_k(weights, k=self.top_k)

        spa_expert_outputs = [expert(x) for expert in self.spatio_experts] 
        expert_outputs = tf.stack(spa_expert_outputs, axis=1)  

        batch_size = tf.shape(x)[0]
        top_k_indices_expanded = tf.reshape(top_k_indices, (batch_size, self.top_k))
        top_k_expert_outputs = tf.gather(expert_outputs, top_k_indices_expanded, batch_dims=1, axis=1)

        # Broadcast the top k weights
        top_k_weights_expanded = tf.reshape(top_k_values, (batch_size, self.top_k, 1))
        top_k_weights_broadcasted = tf.broadcast_to(top_k_weights_expanded, tf.shape(top_k_expert_outputs))

        final_output = tf.reduce_sum(top_k_expert_outputs * top_k_weights_broadcasted, axis=1)
        return self.softmax(final_output)
    
class MoEMLP(layers.Layer):
    def __init__(self, dense_units=64, 
                 gate_units=32, num_classes=6, 
                 num_experts=5, 
                 epsilon=1e-8, top_k=5, dropout_rate=0):
        super(MoEMLP, self).__init__()
        self.spatio_experts = [SpatioExpertMLP(num_classes, dense_units=dense_units) for _ in range(num_experts)]
        self.gating = Gating(gate_units, num_experts)
        self.softmax = layers.Activation('softmax')
        self.epsilon = epsilon
        self.top_k = top_k
        self.dropout_rate = dropout_rate


    def call(self, x):
        weights = self.gating(x)
        
        dropout_mask = tf.nn.dropout(tf.ones_like(weights), rate=self.dropout_rate)
        weights *= dropout_mask

        # Add a small random value to the gating weights
        random_noise = self.epsilon * tf.random.normal((1,))
        weights += random_noise

        top_k_values, top_k_indices = tf.math.top_k(weights, k=self.top_k)

        spa_expert_outputs = [expert(x) for expert in self.spatio_experts] 
        expert_outputs = tf.stack(spa_expert_outputs, axis=1)  

        batch_size = tf.shape(x)[0]
        top_k_indices_expanded = tf.reshape(top_k_indices, (batch_size, self.top_k))
        top_k_expert_outputs = tf.gather(expert_outputs, top_k_indices_expanded, batch_dims=1, axis=1)

        # Broadcast the top k weights
        top_k_weights_expanded = tf.reshape(top_k_values, (batch_size, self.top_k, 1))
        top_k_weights_broadcasted = tf.broadcast_to(top_k_weights_expanded, tf.shape(top_k_expert_outputs))

        final_output = tf.reduce_sum(top_k_expert_outputs * top_k_weights_broadcasted, axis=1)
        return self.softmax(final_output)
        
    
class SqueezeExcitationAttention(layers.Layer):
    def __init__(self, channels, ratio=0.25):
        super().__init__()
        self.squeeze = layers.GlobalAveragePooling2D(keepdims=True, data_format='channels_last')
        self.fc1 = layers.Dense(int(channels*ratio))
        self.relu = layers.Activation('swish')
        self.fc2 = layers.Dense(channels)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, input_block):
        x = self.squeeze(input_block)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        scaling = tf.multiply(input_block, x)
        return scaling

class SEConv(layers.Layer):
    def __init__(self, filters, strides, activation='swish', ratio=0.25):
        super().__init__()
        self.conv1x1 = layers.Conv2D(filters=int(filters*ratio), kernel_size=(1, 1))
        self.batchnorm = layers.BatchNormalization()
        self.activation = layers.Activation(activation)
        self.depthwise = layers.DepthwiseConv2D(kernel_size=(3,3), strides=strides)
        self.batchnorm1 = layers.BatchNormalization()
        self.activation1 = layers.Activation(activation)
        self.att =  SqueezeExcitationAttention(channels=int(filters*ratio))
        self.conv1x1_2 = layers.Conv2D(filters=filters, kernel_size=(1, 1), )
        self.batchnorm3 = layers.BatchNormalization()
        # self.dropout = layers.Dropout(0.1)


    def call(self, input):
        x = self.conv1x1(input)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.depthwise(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)

        x = self.att(x)
        x = self.conv1x1_2(x)
        x = self.batchnorm3(x)
        # x = self.dropout(x)
        return x
    
class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def call(self, y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    
# class DepthwiseConv(layers.Layer):
#     def __init__(self, filters, kernel_size, strides, padding):
#         super(DepthwiseConv, self).__init__()
#         self.dw = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, 
#                                             depth_multiplier=1, padding=padding, 
#                                             use_bias=False)
#         self.bn1 = layers.BatchNormalization()
#         self.act1 = layers.Activation('relu')
        
#         self.pw = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same',
#                                    kernel_regularizer=l2(1e-2), bias_regularizer=l2(1e-2))
#         self.bn2 = layers.BatchNormalization()
#         self.act2 = layers.Activation('relu')
        
#     def call(self, inputs):
#         x = self.dw(inputs)
#         x = self.bn1(x)
#         x = self.act1(x)
#         x = self.pw(x)
#         x = self.bn2(x)
#         x = self.act2(x)
#         return x

# class SpatioExpert(layers.Layer):
#     def __init__(self, num_classes=6, filters = 64, dense_units=32):
#         super(SpatioExpert, self).__init__()
#         self.conv1 = DepthwiseConv(int(filters//4), 3, 2, 'same')
#         self.bn1 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
#         self.act1 = layers.Activation('swish')
#         self.conv2 = DepthwiseConv(int(filters//4), 3, 2, 'same')
#         self.bn2 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
#         self.act2 = layers.Activation('swish')

#         self.conv3 = DepthwiseConv(int(filters//2), 3, 2, 'same')
#         self.bn3 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
#         self.act3 = layers.Activation('swish')
#         self.conv4 = DepthwiseConv(int(filters//2), 3, 2, 'same')
#         self.bn4 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
#         self.act4 = layers.Activation('swish')

#         self.conv5 = DepthwiseConv(filters, 3, 2, 'same')
#         self.bn5 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
#         self.act5 = layers.Activation('swish')
#         self.conv6 = DepthwiseConv(filters, 3, 2, 'same')
#         self.bn6 = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)
#         self.act6 = layers.Activation('swish')

#         self.flatten = layers.Flatten()
#         self.drop = layers.Dropout(0.1)
#         self.fc1 = layers.Dense(dense_units)
#         self.fc2 = layers.Dense(num_classes)

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.act1(x) 
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.act2(x)

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.act3(x)
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.act4(x)

#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.act5(x)
#         x = self.conv6(x)
#         x = self.bn6(x)
#         x = self.act6(x)

#         x = self.flatten(x)
#         x = self.drop(x)
#         x = self.fc1(x)
#         return self.fc2(x)
    
#%%
# num_experts = 10
# top_k = 5
# inputs = tf.keras.Input(shape=(128, 118, 1))
# x = layers.Normalization()(inputs)
# # x = SEConv(8, 2, 'swish')(x)
# # x = SEConv(8, 2, 'swish')(x)
# moes_layer = TemporalExpert(num_classes=6, input_shape= x.shape)

# # Apply the MoEs layer
# outputs = moes_layer(x)
# moes_model = tf.keras.Model(inputs=inputs, outputs=outputs)
# moes_model.summary()
# %%
# python train_submitjob.py --num_experts 100 --top_k 5
# python train_submitjob.py --num_experts 200 --top_k 5
# python train_submitjob.py --num_experts 500 --top_k 5
# %%
