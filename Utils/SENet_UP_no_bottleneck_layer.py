from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    Concatenate,
    GlobalAveragePooling3D, Reshape, Permute)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    Conv3D,
    Conv2D
)
from keras import backend as K

def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4

def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x

def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    # x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
    #                         name=name + '_0_bn')(x)
    # x1 = Activation('relu', name=name + '_0_relu')(x1)
    # x1 = Conv3D(4 * growth_rate, 1, use_bias=False,
    #             name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv3D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv3D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv')(x)
    # x = AveragePooling3D(1, strides=2, name=name + '_pool')(x)
    return x

def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    # print("1" * 10, filters)
    # 160

    # 光谱和空间采样
    se_shape = (1, 1, 1, filters)

    # 空间采样
    # se_shape = (1, 1, filters)

    # Squeeze
    se = GlobalAveragePooling3D()(init)
    # print(se.shape)
    # (?, 96)
    se = Reshape(se_shape)(se)

    # Excitation
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    # print(se.shape)
    # (?, 1, 1, 96)

    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se)

    # 分配权重
    # print(se)
    # Tensor("dense_2/Sigmoid:0", shape=(?, 1, 1, 1, 160), dtype=float32)
    # print(init)
    # Tensor("conv1_block1_concat/concat:0", shape=(?, 5, 5, 99, 96), dtype = float32)

    x = multiply([init, se])
    # print(x.shape)
    # (?, 5, 5, 99, 96)

    return x

# 组合模型
class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_shape)
        # orignal input shape: 1,7,7,200

        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        print('change input shape:', input_shape)

        # 张量流输入
        input = Input(shape=input_shape)

        # 3D Convolution and pooling
        conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', kernel_initializer='he_normal')(
            input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(conv1)

        # Dense Block1
        x = dense_block(pool1, 3, name='conv1')
        x = squeeze_excite_block(x)
        x = transition_block(x, 0.5, name='pool1')
        x = dense_block(x, 3, name='conv2')
        x = squeeze_excite_block(x)
        x = transition_block(x, 0.5, name='pool2')
        x = dense_block(x, 3, name='conv3')
        x = GlobalAveragePooling3D(name='avg_pool')(x)
        # x = Dense(16, activation='softmax')(x)

        # 输入分类器
        # Classifier block
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(x)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        # (1,7,7,200),16
        return ResnetBuilder.build(input_shape, num_outputs)

def main():
    model = ResnetBuilder.build_resnet_8((1, 11, 11, 103), 9)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary(positions=[.33, .61, .71, 1.])

if __name__ == '__main__':
    main()
