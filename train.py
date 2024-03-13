import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

files_train = 0
files_validation = 0

# 获取目录.../park
cwd = os.getcwd()

img_width, img_height = 48, 48
train_data_dir = "train_data/train"
validation_data_dir = "train_data/test"
batch_size = 32
epochs = 15
num_classes = 2

# 使用VGG16模型，载入预训练权重，去掉顶部分类器，设置输入形状
model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# 冻结前10层的权重，不参与训练
for layer in model.layers[:10]:
    layer.trainable = False

# 在VGG16基础上添加全连接层，输出为类别数，并构建最终模型
x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)
model_final = Model(inputs=model.input, outputs=predictions)

# 编译模型，使用随机梯度下降优化器和分类交叉熵损失函数
# accuracy表示模型在训练和验证期间对分类任务的准确预测比例。metrics: 评估指标，用于监控模型的性能
model_final.compile(loss="categorical_crossentropy",
                    optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9),
                    metrics=["accuracy"])

# 图像数据增强生成器，用于训练集和验证集
# 进行图像数据增强，以增加训练数据的多样性
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # 图像的像素缩放到指定的比列
    horizontal_flip=True,  # 随机水平翻转图像
    fill_mode="nearest",  # 用于填充新创建的像素，当进行平移等操作时可能会产生空白区域。"nearest"表示使用最近邻插值来填充。
    zoom_range=0.1,  # 图像有10%的可能性被缩放
    width_shift_range=0.1,  # 随机水平和垂直平移图像的范围
    height_shift_range=0.1,
    rotation_range=5  # 图像有可能被旋转不超过5度
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5)

# 从目录中生成批量训练器，训练和验证数据
# flow_from_directory 方法用于从目录中读取图像数据并进行批量生成。
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),  # 将图像调整为指定的大小
    batch_size=batch_size,  # 设置每个批次的图像数量
    class_mode="categorical"  # 表示标签是独热编码的类别标签，适用于多类别分类任务
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical"
)

# ModelCheckpoint 回调函数，它的主要目的是在训练过程中监视验证集的准确率和保存模型的权重
checkpoint = ModelCheckpoint(
    "car1.h5",  # 保存模型权重的文件路径
    monitor='val_accuracy',  # 选择监视的指标，这里选择了验证集上的准确率。
    verbose=1,  # 设置为1时，会在保存模型时输出一些信息，告诉你模型已经保存
    save_best_only=True,  # 只有当监视的指标在新的 epoch 中取得了更好的结果时才保存模型
    save_weights_only=False,  # 为 True，则只保存模型的权重而不保存整个模型结构。在这里设置为 False，保存整个模型
    mode='auto',  # 在监视的指标上进行模型保存的模式，'auto' 表示自动选择，会根据监视的指标类型来进行判断
    save_freq='epoch'  # 保存模型的频率，这里设置为每个 epoch 保存一次。
)
#  EarlyStopping 回调函数，如果在指定的轮数内准确率没有改善，则提前停止训练
early = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,  # 训练过程中被认为是提升的最小变化，如果新的准确率比之前的提升小于 min_delta，则被认为没有提升。
    patience=10,  # 当连续指定轮数 (patience) 内准确率没有提升时，训练会提前停止
    verbose=1,  # 设置为1时，在提前停止时输出一些信息，告诉你模型已经停止训练。
    mode='auto'
)

# 使用生成器训练模型，并保存训练历史
history_object = model_final.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # 每个 epoch 中的训练步数，即生成器要产生多少个批次数据。这里使用生成器的长度作为训练步数，确保遍历整个训练集
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, early]  # 回调函数列表，包括了在训练过程中使用的
)
