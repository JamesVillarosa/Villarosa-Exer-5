# Villarosa, James Carl V.
# GH4L
# 2022-69578

import tkinter as tk
import os
import re
import random
import matplotlib.pyplot as plt
from tkinter import filedialog
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix


def read_folder(folder_path):
    vocabulary = []
    words_frequency = {}

    dir_list = os.listdir(folder_path)
    for file_name in dir_list:
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='latin-1') as f:
                tokenize = f.read().split()
                regex = re.compile('[^a-zA-Z]')

                for i in tokenize:
                    clean = regex.sub('', i).lower()
                    if clean:
                        vocabulary.append(clean)

    unique_words = set(vocabulary)
    for word in unique_words:
        words_frequency[word] = vocabulary.count(word)

    return vocabulary, words_frequency


def split_data(data, split_ratio=(0.6, 0.2, 0.2)):
    random.shuffle(data)
    train_size = int(split_ratio[0] * len(data))
    val_size = int(split_ratio[1] * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


def calculate_probabilities(bow_spam, bow_ham, classify_files, smoothing_factor):
    spam_total_words = len(bow_spam[0])
    ham_total_words = len(bow_ham[0])
    spam_unique_words = len(bow_spam[1])
    ham_unique_words = len(bow_ham[1])

    spam_word_probabilities = {}
    ham_word_probabilities = {}

    for word in bow_spam[1]:
        word_count = bow_spam[1].get(word, 0)
        spam_word_probabilities[word] = (word_count + smoothing_factor) / (spam_total_words + smoothing_factor * spam_unique_words)

    for word in bow_ham[1]:
        word_count = bow_ham[1].get(word, 0)
        ham_word_probabilities[word] = (word_count + smoothing_factor) / (ham_total_words + smoothing_factor * ham_unique_words)

    classifications = []

    for file_name, clean_words in classify_files:
        spam_prob = 1
        ham_prob = 1

        for word in clean_words:
            spam_prob *= spam_word_probabilities.get(word, smoothing_factor / (spam_total_words + smoothing_factor * spam_unique_words))
            ham_prob *= ham_word_probabilities.get(word, smoothing_factor / (ham_total_words + smoothing_factor * ham_unique_words))

        if spam_prob > ham_prob:
            classifications.append((file_name, "Spam"))
        else:
            classifications.append((file_name, "Ham"))

    return classifications


def evaluate_model(classifications, true_labels):
    y_pred = [label for _, label in classifications]
    accuracy = accuracy_score(true_labels, y_pred)
    precision = precision_score(true_labels, y_pred, pos_label="Spam")
    recall = recall_score(true_labels, y_pred, pos_label="Spam")
    conf_matrix = confusion_matrix(true_labels, y_pred, labels=["Spam", "Ham"])

    TP = conf_matrix[0, 0]
    TN = conf_matrix[1, 1]
    FP = conf_matrix[1, 0]
    FN = conf_matrix[0, 1]

    return accuracy, precision, recall, conf_matrix, TP, TN, FP, FN


def plot_metrics(k_values, accuracies, precisions, recalls):
    plt.plot(k_values, accuracies, label="Accuracy")
    plt.plot(k_values, precisions, label="Precision")
    plt.plot(k_values, recalls, label="Recall")
    plt.xlabel("Laplace Smoothing Factor (k)")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Model Performance vs. Smoothing Factor (k)")
    plt.show()


def classify_and_evaluate(folder_path):
    spam_folder = os.path.join(folder_path, 'spam')
    ham_folder = os.path.join(folder_path, 'ham')

    spam_files = [(file_name, read_clean_words(os.path.join(spam_folder, file_name))) for file_name in os.listdir(spam_folder)]
    ham_files = [(file_name, read_clean_words(os.path.join(ham_folder, file_name))) for file_name in os.listdir(ham_folder)]

    train_spam, val_spam, test_spam = split_data(spam_files)
    train_ham, val_ham, test_ham = split_data(ham_files)

    train_files = train_spam + train_ham
    val_files = val_spam + val_ham
    test_files = test_spam + test_ham

    true_val_labels = ["Spam"] * len(val_spam) + ["Ham"] * len(val_ham)
    true_test_labels = ["Spam"] * len(test_spam) + ["Ham"] * len(test_ham)

    k_values = [0.005, 0.01, 0.5, 1.0, 2.0]
    accuracies, precisions, recalls = [], [], []

    bow_spam = read_folder(spam_folder)
    bow_ham = read_folder(ham_folder)

    for k in k_values:
        classifications = calculate_probabilities(bow_spam, bow_ham, val_files, k)
        accuracy, precision, recall, _, _, _, _, _ = evaluate_model(classifications, true_val_labels)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

    plot_metrics(k_values, accuracies, precisions, recalls)

    best_k = k_values[accuracies.index(max(accuracies))]
    print(f"Best k value: {best_k}")

    final_classifications = calculate_probabilities(bow_spam, bow_ham, test_files, best_k)
    accuracy, precision, recall, conf_matrix, TP, TN, FP, FN = evaluate_model(final_classifications, true_test_labels)

    print(f"Test Accuracy: {accuracy}")
    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")
    print("Confusion Matrix:")
    print(f"[[TP: {TP}, FN: {FN}]")
    print(f" [FP: {FP}, TN: {TN}]]")

def read_clean_words(file_path):
    with open(file_path, 'r', encoding='latin-1') as f:
        content = f.read().split()
        regex = re.compile('[^a-zA-Z]')
        clean_words = [regex.sub('', token).lower() for token in content if regex.sub('', token)]
    return clean_words


def select_folder():
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()
    classify_and_evaluate(folder_path)


select_folder()
