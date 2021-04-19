import sys

from Main import DenseModel
import params

def main(argv):
    baseline = DenseModel(params, argv)
    if baseline.params['train_cnn']:
        # baseline.train_cnn()
        baseline.train_effnet_cnn()
    """
    if baseline.params.generate_cnn_codes:
        baseline.generate_cnn_codes()
    if baseline.params.train_lstm:
        baseline.train_lstm()
    if baseline.params.test_cnn or baseline.params.test_lstm:
        baseline.test_models()
    """

if __name__ == "__main__":
    main(['-all'])
    #main(sys.argv[1:])
