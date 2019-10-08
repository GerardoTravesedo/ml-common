import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools.object_detection as od


class ImageDetectionAnalysis:

    # TODO: Create function that generates recall over precision graph
    # TODO: Create constants for metrics
    # TODO: Add unit test

    def __init__(self, class_labels):
        """
        Metric table represented as a Pandas Dataframe with labels

        TP = iou_over_5_and_same_class
        FP = duplicate + iou_below_5
        FN = iou_over_5_and_diff_class + gt_not_found

        :param class_labels: List containing the names of the different classes that we will use to index the rows in
        the metric table
        """
        column_names = \
            ["iou_over_5_and_same_class", "iou_over_5_and_diff_class", "gt_not_found", "iou_below_5", "duplicate"]
        self.metric_table = pd.DataFrame(0, columns=column_names, index=class_labels)

    def add_record(self, gt_objects, predicted_objects):
        """
        Adds information to the metric table after an object detection prediction in an image. We should call this
        method once per image that we predict.

        This method assumes that the box has format [x1, y1, x2, y2] where (x1, y1) represents the top left corner and
        (x2, y2) represents the bottom right corner.

        :param gt_objects: Ground truth objects containing boxes and classes
        :param predicted_objects: Predicted objects containing boxes and classes
        """
        for predicted_object in predicted_objects:
            max_iou = 0
            max_iou_gt_object = None
            metric = None

            for gt_object in gt_objects:
                iou = od.calculate_iou_two_points(predicted_object["bbox"], gt_object["bbox"])

                if predicted_object["class"] == gt_object["class"] and iou >= 0.5 and "matched" not in gt_object:
                    if metric == "iou_over_5_and_same_class" and iou > max_iou:
                        max_iou = iou
                        max_iou_gt_object = gt_object
                    elif metric != "iou_over_5_and_same_class":
                        max_iou_gt_object = gt_object
                        max_iou = iou
                        metric = "iou_over_5_and_same_class"
                elif predicted_object["class"] == gt_object["class"] and iou >= 0.5 \
                        and "matched" in gt_object and metric != "iou_over_5_and_same_class":
                    metric = "duplicate"
                elif predicted_object["class"] != gt_object["class"] and iou >= 0.5 \
                        and metric != "iou_over_5_and_same_class" and metric != "duplicate":
                    metric = "iou_over_5_and_diff_class"

            if metric == "iou_over_5_and_same_class":
                max_iou_gt_object["matched"] = True
            elif not metric:
                metric = "iou_below_5"

            self.metric_table.loc[predicted_object["class"], metric] += 1

        for gt_object in gt_objects:
            if "matched" not in gt_object:
                self.metric_table.loc[gt_object["class"], "gt_not_found"] += 1

    def get_class_precision(self, class_name):
        """
        Precision = true positive / (true positive + false positive)

        Use it when the cost of false positives is high. For example in spam detection. A false positive means
        that an email that wan't spam was categorized as spam. The user might miss important emails.

        :param class_name: Class name to calculate the precision for
        :return: precision of class class_name
        """
        true_positive = self.metric_table.loc[class_name, "iou_over_5_and_same_class"]
        false_positive = self.metric_table.loc[class_name, "duplicate"] + \
                         self.metric_table.loc[class_name, "iou_below_5"]
        return round(true_positive / (true_positive + false_positive), 3)

    def get_class_recall(self, class_name):
        """
        Recall = true positive / (true positive + false negative)

        Use it when the cost of false negatives is high. For example, telling people with cancer that they don't have it
        can be very bad.

        :param class_name: Class name to calculate the recall for
        :return: recall of class class_name
        """
        true_positive = self.metric_table.loc[class_name, class_name]
        false_negative = self.metric_table.loc[class_name, "gt_not_found"] + \
                         self.metric_table.loc[class_name, "iou_over_5_and_diff_class"]
        return round(true_positive / (true_positive + false_negative), 3)

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

    def generate_metric_table_heat_map(self, output_path):
        """
        :param output_path: Path to folder where image containing heat map will be saved
        :return: Saves metric table heat map into a file
        """
        class_labels = list(self.metric_table.index.values)
        column_labels = list(self.metric_table.column.values)
        numpy_matrix = self.metric_table.values

        fig, ax = plt.subplots()

        # We use hot_r which is the hot colormap but reversed
        # White is the lowest, black is the highest
        ax.imshow(numpy_matrix, cmap='hot_r')
        ax.set_xticks(np.arange(len(column_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(column_labels)
        ax.set_yticklabels(class_labels)

        for row in range(len(class_labels)):
            for col in range(len(column_labels)):
                ax.text(col, row, numpy_matrix[row, col], ha="center", va="center", color="g")

        # plt.show()
        fig.savefig(output_path + "/metric-table.png")
