import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, Lambda
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score


def load_data(subject_id):
    data = []
    labels = []
    for session_id in range(1, 4):
        data_path = f'C:/Users/123/OneDrive/桌面/4J大作业/SEED-IV/{session_id}/{subject_id}'
        train_data = np.load(os.path.join(data_path, 'train_data.npy'))
        train_labels = np.load(os.path.join(data_path, 'train_label.npy'))
        test_data = np.load(os.path.join(data_path, 'test_data.npy'))
        test_labels = np.load(os.path.join(data_path, 'test_label.npy'))

        data.append(np.concatenate([train_data, test_data], axis=0))
        labels.append(np.concatenate([train_labels, test_labels], axis=0))

    return np.concatenate(data, axis=0), np.concatenate(labels, axis=0)


def gradient_reversal(x, hp_lambda):
    return tf.multiply(x, -hp_lambda)


def create_domain_adaptation_model(input_shape, num_classes, hp_lambda):
    inputs = Input(shape=input_shape)

    feature_extractor = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs)
    feature_extractor = Conv1D(filters=32, kernel_size=3, activation='relu')(feature_extractor)
    feature_extractor = GlobalAveragePooling1D()(feature_extractor)

    grl = Lambda(gradient_reversal, arguments={'hp_lambda': hp_lambda})(feature_extractor)

    domain_classifier = Dense(32, activation='relu')(grl)
    domain_classifier = Dense(1, activation='sigmoid', name='domain_output')(domain_classifier)

    emotion_classifier = Dense(num_classes, activation='softmax', name='emotion_output')(feature_extractor)

    model = Model(inputs=inputs, outputs=[emotion_classifier, domain_classifier])
    return model


def domain_adaptation(train_data, train_labels, test_data, test_labels, hp_lambda):
    input_shape = train_data.shape[1:]
    num_classes = len(np.unique(train_labels))

    model = create_domain_adaptation_model(input_shape, num_classes, hp_lambda)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'emotion_output': 'sparse_categorical_crossentropy',
                        'domain_output': 'binary_crossentropy'},
                  metrics={'emotion_output': 'accuracy',
                           'domain_output': 'accuracy'})

    domain_labels = np.concatenate([np.ones(train_data.shape[0]), np.zeros(test_data.shape[0])], axis=0)
    mixed_data = np.concatenate([train_data, test_data], axis=0)

    model.fit(mixed_data, {'emotion_output': np.concatenate([train_labels, test_labels], axis=0),
                           'domain_output': domain_labels},
              batch_size=32, epochs=50, verbose=2)

    test_preds = np.argmax(model.predict(test_data)[0], axis=-1)
    accuracy = accuracy_score(test_labels, test_preds)

    return accuracy


def main():
    hp_lambda = 0.5
    accuracies = []

    for test_subject_id in range(1, 16):
        train_data, train_labels = [], []
        for train_subject_id in range(1, 16):
            if train_subject_id == test_subject_id:
                continue

            data, labels = load_data(train_subject_id)
            train_data.append(data)
            train_labels.append(labels)

        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        test_data, test_labels = load_data(test_subject_id)

        accuracy = domain_adaptation(train_data, train_labels, test_data, test_labels, hp_lambda)
        print(f'Test Subject {test_subject_id} Accuracy: {accuracy:.4f}')

        accuracies.append(accuracy)

    print(f'Average Accuracy: {np.mean(accuracies):.4f}')


if __name__ == '__main__':
    main()