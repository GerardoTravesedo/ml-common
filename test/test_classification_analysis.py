import classification_analysis as ca


class TestClassificationAnalysis(object):

    def test_precision(self):
        valid_classes = ['Person', 'Cat', 'Dog']
        analyzer = ca.ClassificationAnalysis(valid_classes)

        analyzer.add_record('Person', 'Person')
        analyzer.add_record('Person', 'Person')
        analyzer.add_record('Cat', 'Cat')
        analyzer.add_record('Cat', 'Cat')
        analyzer.add_record('Dog', 'Dog')
        analyzer.add_record('Dog', 'Dog')
        analyzer.add_record('Dog', 'Dog')
        analyzer.add_record('Cat', 'Dog')
        analyzer.add_record('Cat', 'Dog')
        analyzer.add_record('Cat', 'Dog')
        analyzer.add_record('Dog', 'Cat')
        analyzer.add_record('Dog', 'Cat')
        analyzer.add_record('Dog', 'Cat')

        assert analyzer.get_class_precision('Person') == 1
        assert analyzer.get_class_recall('Person') == 1

        assert analyzer.get_class_precision('Cat') == 0.4
        assert analyzer.get_class_recall('Cat') == 0.4

        assert analyzer.get_class_precision('Dog') == 0.5
        assert analyzer.get_class_recall('Dog') == 0.5

        assert analyzer.get_class_f1('Person') == 1
        assert analyzer.get_class_f1('Cat') == 0.4
        assert analyzer.get_class_f1('Dog') == 0.5
