import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense
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

        # 添加更多的特征
        train_data = np.concatenate([train_data, np.square(train_data), np.log(train_data)], axis=-1)
        test_data = np.concatenate([test_data, np.square(test_data), np.log(test_data)], axis=-1)

        data.append(np.concatenate([train_data, test_data], axis=0))
        labels.append(np.concatenate([train_labels, test_labels], axis=0))

    return np.concatenate(data, axis=0), np.concatenate(labels, axis=0)


def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def transfer_learning(train_data, train_labels, test_data, test_labels):
    input_shape = train_data.shape[1:]
    num_classes = len(np.unique(train_labels))
    model = create_cnn_model(input_shape, num_classes)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels, batch_size=40, epochs=30, verbose=2)

    test_preds = np.argmax(model.predict(test_data), axis=-1)
    accuracy = accuracy_score(test_labels, test_preds)

    return accuracy


def main():
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

        accuracy = transfer_learning(train_data, train_labels, test_data, test_labels)
        print(f'Test Subject {test_subject_id} Accuracy: {accuracy:.4f}')

        accuracies.append(accuracy)

    print(f'Average Accuracy: {np.mean(accuracies):.4f}')


if __name__ == '__main__':
    main()
