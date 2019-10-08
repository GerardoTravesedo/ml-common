import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ClassificationAnalysis:

    def __init__(self, class_labels):
        """
        Confusion matrix represented as a Pandas Dataframe with labels

        :param class_labels: List containing the names of the different classes that we will use to index the
        confusion matrix
        """
        self.confusion_matrix = pd.DataFrame(0, columns=class_labels, index=class_labels)

    def add_record(self, gt_class, predicted_class):
        """
        Use this method after a classification prediction using the ground truth class and the predicted one. It
        updates the confusion matrix with the information provided.

        :param gt_class: Name of the ground truth class
        :param predicted_class: Name of the predicted class
        """
        self.confusion_matrix.loc[gt_class, predicted_class] += 1

    def add_records(self, gt_classes, predicted_classes):
        """
        Updates confusion matrix with the provided pairs of ground truth and predicted classes. The order of elements
        in gt_classes and predicted_classes matters. Element in index i if gt_classes must be the ground truth class
        for the predicted class in index i of predicted_classes.

        :param gt_classes: List of ground truth classes
        :param predicted_classes: List of predicted classes
        """
        for gt_class, predicted_class in zip(gt_classes, predicted_classes):
            self.confusion_matrix.loc[gt_class, predicted_class] += 1

    def get_class_precision(self, class_name):
        """
        Precision = true positive / (true positive + false positive)
        Among all the samples that were predicted as class_name, what percentage was actually correct

        Use it when the cost of false positives is high. For example in spam detection. A false positive means
        that an email that wan't spam was categorized as spam. The user might miss important emails.

        :param class_name: Class name to calculate the precision for
        :return: precision of class class_name
        """
        true_positive = self.confusion_matrix.loc[class_name, class_name]
        total_positive = self.confusion_matrix.loc[:, class_name].sum(axis=0)
        return round(true_positive / total_positive, 3)

    def get_class_recall(self, class_name):
        """
        Recall = true positive / (true positive + false negative)
        Among all samples that should have been predicted as this class, what percentage were correctly classified

        Use it when the cost of false negatives is high. For example, telling people with cancer that they don't have it
        can be very bad.

        :param class_name: Class name to calculate the recall for
        :return: recall of class class_name
        """
        true_positive = self.confusion_matrix.loc[class_name, class_name]
        total = self.confusion_matrix.loc[class_name, :].sum(axis=0)
        return round(true_positive / total, 3)

    def get_class_f1(self, class_name):
        """
        F1 = 2 * (precision * recall / (precision + recall))
        Use it when you wan't to find a balance between precision and recall

        :param class_name: Class name to calculate F1 for
        :return: F1 of class class_name
        """
        precision = self.get_class_precision(class_name)
        recall = self.get_class_recall(class_name)
        return round((2 * precision * recall) / (precision + recall), 3)

    def generate_confusion_matrix_heat_map(self, output_path):
        """
        :param output_path: Path to folder where image containing heat map will be saved
        :return: Saves confusion matrix heat map into a file
        """
        labels = list(self.confusion_matrix.columns.values)
        numpy_matrix = self.confusion_matrix.values

        fig, ax = plt.subplots()

        # We use hot_r which is the hot colormap but reversed
        # White is the lowest, black is the highest
        ax.imshow(numpy_matrix, cmap='hot_r')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        for row in range(len(labels)):
            for col in range(len(labels)):
                ax.text(col, row, numpy_matrix[row, col], ha="center", va="center", color="g")

        # plt.show()
        fig.savefig(output_path + "/confusion-matrix.png")

