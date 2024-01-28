import keras
import keras.backend as tf
import tensorflow
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D
from keras.layers.convolutional import AveragePooling2D, Conv2D, DepthwiseConv2D, SeparableConv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.constraints import max_norm
import numpy as np
from pro_loss import pro_loss1
keras.backend.set_image_data_format('channels_last')
import math

class PSAT:
    def __init__(self, chans, samples, n_output, n_nuisance, lambda_value, architecture='EEGNet', adversarial=False, lam=0.1):

        # Input, data set and model training scheme parameters
        self.chans = chans
        self.samples = samples
        self.n_output = n_output
        self.n_nuisance = n_nuisance
        self.lam = lam

        # Build the network blocks
        self.enc = self.encoder_model(architecture)
        self.latent_dim = self.enc.output_shape[-1]  # inherit latent dimensionality
        self.cla = self.classifier_model(architecture)
        self.adv = self.adversary_model()

        #prototype
        self.pro1 = self.prototype_model1()
        self.latent_dim1 = self.pro1.output_shape[-1]
        self.pro2 = self.prototype_model2()

        # Compile the network with or without adversarial censoring
        input = Input(shape=(self.chans, self.samples, 1))
        latent = self.enc(input)
        output = self.cla(latent)
        leakage = self.adv(latent)

        out2 = self.pro1(latent)
        pro_cls = self.pro2(out2)

        self.adv.trainable = False
        self.acnn = Model(input, [output, leakage, out2])


        if adversarial:
            self.acnn.compile(loss=[lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True),
                                    lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True),
                                    pro_loss1],
                              loss_weights=[1., -1. * self.lam, lambda_value], optimizer=Adam(lr=1e-3, decay=1e-4),
                              metrics=['accuracy'])

        else:   # trains a regular (non-adversarial) CNN, but will monitor leakage via the adversary alongside
            self.acnn.compile(loss=[lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True),
                                    lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True)],
                              loss_weights=[1., 0.], optimizer=Adam(lr=1e-3, decay=1e-4),
                              metrics=['accuracy'])

        self.adv.trainable = True
        self.adv.compile(loss=lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True),
                         loss_weights=[self.lam],
                         optimizer=Adam(lr=1e-3, decay=1e-4),
                         metrics=['accuracy'])

        self.pro1.trainable = True
        self.pro1.compile(loss=pro_loss1, loss_weights=[1.],
                         optimizer=Adam(lr=1e-3, decay=1e-4),
                         metrics=['accuracy'])

        self.pro1.trainable = True
        self.pro2.compile(loss=lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True),
                              loss_weights=[1.], optimizer=Adam(lr=1e-3, decay=1e-4),
                              metrics=['accuracy'])


    def encoder_model(self, architecture):
        model = Sequential()
        if architecture == 'EEGNet':
            model.add(Conv2D(8, (1, 32), padding='same', use_bias=False))
            model.add(BatchNormalization(axis=3))
            model.add(DepthwiseConv2D((self.chans, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.)))
            model.add(BatchNormalization(axis=3))
            model.add(Activation('elu'))
            model.add(AveragePooling2D((1, 4)))
            model.add(Dropout(0.25))
            model.add(SeparableConv2D(16, (1, 16), use_bias=False, padding='same'))
            model.add(BatchNormalization(axis=3))
            model.add(Activation('elu'))
            model.add(AveragePooling2D((1, 8)))
            model.add(Dropout(0.25))
            model.add(Flatten())
        elif architecture == 'DeepConvNet':
            model.add(Conv2D(25, (1, 5)))
            model.add(Conv2D(25, (self.chans, 1), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(50, (1, 5), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(100, (1, 5), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(200, (1, 5), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Flatten())
        elif architecture == 'ShallowConvNet':
            model.add(Conv2D(40, (1, 13)))
            model.add(Conv2D(40, (self.chans, 1), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation(lambda x: tf.square(x)))
            model.add(AveragePooling2D(pool_size=(1, 35), strides=(1, 7)))
            model.add(Activation(lambda x: tf.log(tf.clip(x, min_value=1e-7, max_value=10000))))
            model.add(Dropout(0.5))
            model.add(Flatten())

        input = Input(shape=(self.chans, self.samples, 1))
        latent = model(input)

        return Model(input, latent, name='enc')

    def classifier_model(self, architecture):
        latent = Input(shape=(self.latent_dim,))
        if architecture == 'EEGNet':
            output = Dense(self.n_output, kernel_constraint=max_norm(0.25))(latent)
        elif architecture == 'DeepConvNet' or architecture == 'ShallowConvNet':
            output = Dense(self.n_output)(latent)

        return Model(latent, output, name='cla')

    def adversary_model(self):
        latent = Input(shape=(self.latent_dim,))
        leakage = Dense(self.n_nuisance)(latent)

        return Model(latent, leakage, name='adv')

    #prototype network
    def prototype_model1(self):
        latent = Input(shape=(self.latent_dim,))
        out = Dense(128, name='protoype_net1_out')(latent)
        return Model(latent, out, name='pro1')

    def prototype_model2(self):
        latent = Input(shape=(self.latent_dim1,))
        out = Dense(2, name='protoype_net1_out')(latent)
        return Model(latent, out, name='pro2')

    def train(self, train_set, val_set, result_log, weights_log, epochs=100, batch_size=4):
        #To use the train function, both train_set and validation_set should be three-element tuples (i.e. x_train, y_train, s_train = train_set).
        #Here, the first element x_train is EEG data of size (num_observations, num_channels, num_timesamples, 1),
        #y_train is the one-hot encoded class label (e.g. for a binary label it would have size (num_observations, 2)),
        #And s_train is the one-hot encoded interference label (e.g. for class 10 the interference label will have size (num_observations, 10)).
        # The variable log represents the directory string where log files are saved during training.

        x_train, y_train, s_train = train_set
        x_test, y_test, s_test = val_set

        train_index = np.arange(y_train.shape[0])
        train_batches = [(i * batch_size, min(y_train.shape[0], (i + 1) * batch_size))
                         for i in range((y_train.shape[0] + batch_size - 1) // batch_size)]

        max_acc = 0
        for epoch in range(1, epochs + 1):
            print('Epoch {}/{}'.format(epoch, epochs))
            np.random.shuffle(train_index)
            train_log = []
            prototype1_log = []
            prototype2_log = []
            adv_log = []
            val_log = []
            for iter, (batch_start, batch_end) in enumerate(train_batches):
                batch_ids = train_index[batch_start:batch_end]
                x_train_batch = x_train[batch_ids]
                y_train_batch = y_train[batch_ids]
                s_train_batch = s_train[batch_ids]
                z_train_batch = self.enc.predict_on_batch(x_train_batch)    #Return the features rolled out for each batch of samples

                #domain classifier
                adv_log.append(self.adv.train_on_batch(z_train_batch, s_train_batch))
                #prototype
                pro_fc1 = self.pro1.predict_on_batch(z_train_batch)
                a = np.zeros((batch_size-2, 128))
                w = self.pro2.get_weights()[0].T
                w = np.concatenate((w, a), axis=0)
                w1 = self.pro1.get_weights()[0].T
                bb = np.argmax(y_train_batch, 1)
                w[2, 0:batch_size] = bb
                prototype1_log.append(self.pro1.train_on_batch(z_train_batch, w))
                prototype2_log.append(self.pro2.train_on_batch(pro_fc1, y_train_batch))

                #Classifier optimization
                train_log.append(self.acnn.train_on_batch(x_train_batch, [y_train_batch, s_train_batch, w]))

                # Prototype loss optimization
                if batch_start<np.size(y_test,0):
                    x_test_batch = x_test[batch_start:batch_end]
                    y_test_batch = y_test[batch_start:batch_end]
                    s_test_batch = s_test[batch_start:batch_end]
                    a = np.zeros((batch_size - 2, 128))
                    w = self.pro2.get_weights()[0].T
                    w = np.concatenate((w, a), axis=0)
                    bb = np.argmax(y_test_batch, 1)
                    w[2, 0:batch_size] = bb
                    val_log.append(self.acnn.test_on_batch(x_test_batch, [y_test_batch, s_test_batch, w]))

            train_log = np.mean(train_log, axis=0)
            val_log = np.mean(val_log, axis=0)

            #If the cla of the current target domain val is greater than the previous cla, save the weight
            if val_log[4] > max_acc:
                self.acnn.save_weights(weights_log+'.h5')
                max_acc = val_log[4]

            # Logging model training information per epoch
            print("Train - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%] - [PRO loss: %f, acc: %.2f%%]"
                  % (train_log[0], train_log[1], 100*train_log[4], train_log[2], 100*train_log[5], train_log[3], train_log[6]))
            print("Validation - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%] - [PRO loss: %f, acc: %.2f%%]"
                  % (val_log[0], val_log[1], 100*val_log[4], val_log[2], 100*val_log[5], val_log[3], val_log[6]))
            with open(result_log + '_train.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(train_log[0]) + ',' + str(train_log[1]) + ',' +
                        str(train_log[3]) + ',' + str(train_log[2]) + ',' + str(100*train_log[4]) + '\n')
            with open(result_log + '_validation.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(val_log[0]) + ',' + str(val_log[1]) + ',' +
                        str(val_log[3]) + ',' + str(val_log[2]) + ',' + str(100*val_log[4]) + '\n')

    def cls_loss_result1(self, tar_val_set):
        #separate verification method
        x_tar_test, y_tar_test, s_tar_test = tar_val_set
        batch_size = 4
        cls_test_batches = [(i * batch_size, min(y_tar_test.shape[0], (i + 1) * batch_size))
                            for i in range((y_tar_test.shape[0] + batch_size - 1) // batch_size)]
        target_val_log = []
        for iter, (batch_start, batch_end) in enumerate(cls_test_batches):

            test_iter_num = math.ceil(len(x_tar_test) / batch_size)
            test_iter = iter % test_iter_num
            x_test_batch = x_tar_test[test_iter * batch_size:(test_iter + 1) * batch_size]
            y_test_batch = y_tar_test[test_iter * batch_size:(test_iter + 1) * batch_size]
            s_test_batch = s_tar_test[test_iter * batch_size:(test_iter + 1) * batch_size]
            if test_iter == test_iter_num - 1:
                queshaode = test_iter_num * batch_size - len(x_tar_test)
                queshaode_x = x_tar_test[0:queshaode]
                queshaode_y = y_tar_test[0:queshaode]
                queshaode_s = s_tar_test[0:queshaode]
                x_test_batch = np.concatenate((x_test_batch, queshaode_x), axis=0)
                y_test_batch = np.concatenate((y_test_batch, queshaode_y), axis=0)
                s_test_batch = np.concatenate((s_test_batch, queshaode_s), axis=0)
            a = np.zeros((batch_size - 2, 128))
            w = self.pro2.get_weights()[0].T
            w = np.concatenate((w, a), axis=0)
            bb = np.argmax(y_test_batch, 1)
            w[2, 0:batch_size] = bb
            target_val_log.append(self.acnn.test_on_batch(x_test_batch, [y_test_batch, s_test_batch, w]))

        target_val_log = np.mean(target_val_log, axis=0)
        return target_val_log

    def load_weight(self, weights_path):
        self.acnn.load_weights(weights_path)

    def sigmoid_result(self, x_test):
        return self.cla.predict_on_batch(self.enc.predict_on_batch(x_test))

    def extract_features(self, x):
        return self.enc.predict_on_batch(x);
