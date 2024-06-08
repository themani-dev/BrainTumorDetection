import os
import warnings
import glob
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import Input, Model
from keras import layers
import cv2
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.models import load_model
from keras.metrics import MeanMetricWrapper

# Main class to initialize and setup basic config
class DeepLabV3:
    def __init__(self,DATA_ROOT,EPOCHS):
        self.image_paths = []
        self.DATA_ROOT = DATA_ROOT
        self.IMAGE_SIZE = (128, 128)
        self.EPOCHS =EPOCHS
        self.X = None
        self.Y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.model = None
        self.config()
        self.seed()

    # Function to setup basic UI configuration
    def config(self):
        sns.set_style('darkgrid')
        warnings.simplefilter(action='ignore', category=FutureWarning)

    # Seeding to replicate results in any machine since we need samoling further
    def seed(self):
        import random
        random.seed(42) # Seeding to 42 by default. Can be changed to any number
        np.random.seed(42)
        tf.random.set_seed(42)
        os.environ['PYTHONHASHSEED'] = str(42)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # Main DeeplabV3 model which has complete architecture
    def create_model_DeepLabV3(self,X_shape, classes=1, name="DeepLabV3"):
        # Convolution block
        def conv_block(x, *, filters, kernel_size=3, strides=1, dilation_rate=1, use_bias=False, padding='same',
                       activation='relu', name=""):
            x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
                              use_bias=use_bias, padding=padding, kernel_initializer="he_normal", name=f"{name}_conv")(
                x)
            x = layers.BatchNormalization(name=f"{name}_norm")(x)
            if activation:
                x = layers.Activation(activation, name=f"{name}_acti")(x)
            return x

        # Encoder Block
        def encoder_block(x, *, filters, name="", pooling=True):
            for i, f in enumerate(filters):
                x = conv_block(x, filters=f, name=f'{name}_block{i}')
            if pooling:
                x = layers.MaxPooling2D((2, 2), name=f'{name}_maxpool')(x)
            return x
        # Atrous Spatial Pyramid Pooling Block
        def aspp_block(x, *, filters, name=""):
            dims = x.shape

            out_pool = layers.AveragePooling2D(pool_size=dims[-3:-1], name=f"{name}_avrg_pool")(x)
            out_pool = conv_block(out_pool, filters=filters, kernel_size=1, use_bias=True, name=f"{name}_conv1")
            out_pool = layers.UpSampling2D(size=dims[-3:-1], interpolation="bilinear", name=f"{name}_upsampl")(out_pool)

            out_1 = conv_block(x, filters=filters, kernel_size=1, dilation_rate=1, name=f"{name}_conv2")
            out_4 = conv_block(x, filters=filters, kernel_size=3, dilation_rate=4, name=f"{name}_conv3")
            out_8 = conv_block(x, filters=filters, kernel_size=3, dilation_rate=8, name=f"{name}_conv4")

            x = layers.Concatenate(axis=-1, name=f"{name}_concat")([out_pool, out_1, out_4, out_8])
            output = conv_block(x, filters=filters, kernel_size=1, name=f"{name}_conv5")
            return output

        inputs = Input(X_shape[-3:], name='inputs')
        # Stacking encoder blocks
        x1 = encoder_block(inputs, filters=(32, 32), name="enc_1")
        x2 = encoder_block(x1, filters=(64, 64), name="enc_2")
        x3 = encoder_block(x2, filters=(128, 128), name="enc_3")
        # upsampling ASPP block
        aspp = aspp_block(x3, filters=256, name="aspp")
        dec_input_a = layers.UpSampling2D(
            size=(self.IMAGE_SIZE[0] // aspp.shape[-3] // 2, self.IMAGE_SIZE[1] // aspp.shape[-2] // 2),
            interpolation="bilinear", name="dec_input_a")(aspp)
        dec_input_b = conv_block(x1, filters=64, kernel_size=1, name="dec_input_b")

        # Merging encoder blocks and ASPP blocks
        x = layers.Concatenate(axis=-1, name="dec_concat")([dec_input_a, dec_input_b])
        x = conv_block(x, filters=128, kernel_size=3, name=f"dec_conv")
        x = layers.UpSampling2D(size=(self.IMAGE_SIZE[0] // x.shape[-3], self.IMAGE_SIZE[1] // x.shape[-2]),
                                interpolation="bilinear",
                                name="dec_output")(x)

        # Output
        outputs = conv_block(x, filters=classes, kernel_size=(1, 1), activation='sigmoid', name="outputs")

        return Model(inputs=inputs, outputs=outputs, name=name)

    # DICE Similarity coefficient functions
    def dsc(self,y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score

    # DICE Loss Function
    def dice_loss(self,y_true, y_pred):
        loss = 1 - self.dsc(y_true, y_pred)
        return loss

    # Binary Crossentropy + Dice Loss Function:
    def bce_dice_loss(self,y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss

    # Jaccard similarity function
    def jaccard_similarity(self,y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f) + smooth
        union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f)) + smooth
        return intersection / union

    def jaccard_loss(self,y_true, y_pred):
        return 1 - self.jaccard_similarity(y_true, y_pred)

    def LoadImageData(self):
        def get_image_data(image_paths):
            x, y = list(), list()
            for image_path, mask_path in image_paths:
                image = cv2.imread(os.path.join(self.DATA_ROOT, image_path), flags=cv2.IMREAD_COLOR)
                image = cv2.resize(image, self.IMAGE_SIZE)
                mask = cv2.imread(os.path.join(self.DATA_ROOT, mask_path), flags=cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, self.IMAGE_SIZE)
                x.append(image)
                y.append(mask)
            return np.array(x) / 255, np.expand_dims(np.array(y) / 255, -1)

        for path in glob.glob(self.DATA_ROOT + "**/*_mask.tif"):

            def strip_base(p):
                parts = pathlib.Path(p).parts
                return os.path.join(*parts[-2:])

            image = path.replace("_mask", "")
            if os.path.isfile(image):
                self.image_paths.append((strip_base(image), strip_base(path)))
            else:
                print("MISSING: ", image, "==>", path)

        rows, cols = 3, 3
        fig = plt.figure(figsize=(12, 12))
        for i in range(1, rows * cols + 1):
            fig.add_subplot(rows, cols, i)
            img_path, mask_path = self.image_paths[i]
            img = cv2.imread(self.DATA_ROOT + img_path, flags=cv2.IMREAD_COLOR)
            img = cv2.resize(img, self.IMAGE_SIZE)
            msk = cv2.imread(self.DATA_ROOT + mask_path, flags=cv2.IMREAD_GRAYSCALE)
            msk = cv2.resize(msk, self.IMAGE_SIZE)
            plt.imshow(img)
            plt.imshow(msk, alpha=0.4)
        plt.show()
        plt.savefig("./output/InitialMask.png")

        self.X, self.Y = get_image_data(self.image_paths)

        print(f"X: {self.X.shape}")
        print(f"Y: {self.Y.shape}")

    def TestTrainSplit(self,size):
        ### Deep Learnv3
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=size)

        print(f"x_train: {self.x_train.shape}, y_train: {self.y_train.shape}")
        print(f"x_test:  {self.x_test.shape},  y_test:  {self.y_test.shape}")

    def ModelInitialization(self):
        self.model = self.create_model_DeepLabV3(self.x_test.shape, 1)
        self.model.compile(optimizer="adam", loss=self.bce_dice_loss, metrics=[self.dsc, self.jaccard_similarity])
        self.model.summary()

    def ModelTraining(self):
        MODEL_CHECKPOINT = f"./working/model/{self.model.name}.ckpt"
        callbacks_list = [
            keras.callbacks.EarlyStopping(monitor='val_dsc', mode='max', patience=20),
            keras.callbacks.ModelCheckpoint(filepath=MODEL_CHECKPOINT, monitor='val_dsc', save_best_only=True,
                                            mode='max', verbose=1)
        ]
        history = self.model.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=self.EPOCHS,
            callbacks=callbacks_list,
            validation_split=0.2,
            verbose=1)

        # Saving model to HDF5 format file to reload in accuracy finding
        self.model.save(f"./compiled/{self.model.name}.hd5")
        print("Model Fitted and saved under compiled folder in output")

        # plot to show the trainig history of Neural Network
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))
        sns.lineplot(data={k: history.history[k] for k in ('loss', 'val_loss')}, ax=ax[0])
        sns.lineplot(data={k: history.history[k] for k in history.history.keys() if k not in ('loss', 'val_loss')},
                     ax=ax[1])
        plt.show()
        plt.savefig("./output/TrainingHistory.png")
        return history
    def ModelLoad(self):
        dsc_metric = MeanMetricWrapper(fn=self.dsc,name='dsc')
        self.model  = load_model(filepath=f"./compiled/DeepLabV3.hd5", custom_objects={'MeanMetricWrapper': dsc_metric})
        print(self.model.name)

    def ModelTest(self):

        self.y_pred = self.model.predict(self.x_test)
        self.y_pred = (self.y_pred > 0.5).astype(np.float64)

        for _ in range(20):
            i = np.random.randint(len(self.y_test))
            if self.y_test[i].sum() > 0:
                plt.figure(figsize=(8, 8))
                plt.subplot(1, 3, 1)
                plt.imshow(self.x_test[i])
                plt.title('Original Image')
                plt.subplot(1, 3, 2)
                plt.imshow(self.y_test[i])
                plt.title('Original Mask')
                plt.subplot(1, 3, 3)
                plt.imshow(self.y_pred[i])
                plt.title('Prediction')
                plt.show()
                plt.savefig(f"./output/Prediction_{_}.png")

    def ModelMetrics(self):

        pred_dice_metric = np.array([self.dsc(self.y_test[i], self.y_pred[i]).numpy() for i in range(len(self.y_test))])
        fig = plt.figure(figsize=(8, 4))
        sns.histplot(pred_dice_metric, stat="probability", bins=50)
        plt.xlabel("Dice metric")
        plt.savefig("./output/Dice.png")
        plt.show()


        pred_jaccard_metric = np.array([self.jaccard_similarity(self.y_test[i], self.y_pred[i]).numpy() for i in range(len(self.y_test))])

        fig = plt.figure(figsize=(8, 4))
        sns.histplot(pred_jaccard_metric, stat="probability", bins=50)
        plt.xlabel("Jaccard (IoU) metric")
        plt.savefig("./output/Jaccard.png")
        plt.show()

