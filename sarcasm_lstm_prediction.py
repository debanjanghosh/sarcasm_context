# structure based off of https://github.com/chridey/cmv/blob/master/cmv/bin/cmv_predict_rnn.py
import os
os.environ["THEANO_FLAGS"] = "floatX=float32"
import sys

from lstm.release.sarcasmClassifier import SarcasmClassifier

from lstm.getData import load_data
from lstm.getData import load_twitter_wsd_data


from com.ccls.lstm.preprocess.process_properties import PreProcessor
#from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime

#from sklearn.mo
from com.ccls.lstm.preprocess.preprocessing_input import listTargets
def main():
    ''' this is for WSD'''
    targets = listTargets()
    targets = ['always']
    
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_file = open("./data/output/logs/log_file_{}".format(time_stamp), "w+")

    recurrent_dimensions = [80,100,150]
    patiences = [10]
    dropouts = [0.25,0.5,0.75]
    lambda_ws = [.0000001,.000001,.00001,.0001]
    processor = PreProcessor(sys.argv[1])

    for target in targets:
        for lambda_w in lambda_ws:
            for dropout in dropouts:
                for recurrent_dimension in recurrent_dimensions:
                    for patience in patiences:
                        if processor.test_type  == "train_test_twitter":
                            
                            print("working on target: {}\n".format(target))
                            log_file.write("working on target: {} at {} with lambda_w: {} dropout: {} recurrent_dimension: {} patience: {}\n".format(target, time_stamp,lambda_w, dropout, recurrent_dimension, patience))

                            processor.set_target(target)
                            training, y, testing, test_y, kwargs = load_twitter_wsd_data(processor)
                            kwargs.update({'lambda_w' : lambda_w, 'dropout': dropout, "num_hidden": recurrent_dimension, "patience": patience})
                            classifier = SarcasmClassifier(**kwargs)
                            classifier.fit(training, y, log_file)
                            classifier.save('./data/output/models/classifier_{}_{}_{}_{}_{}_{}'.format(target, time_stamp,lambda_w, dropout, recurrent_dimension, patience))
                            preds,scores = classifier.predict(testing, test_y)
                            precision, recall, fscore = scores[0], scores[1], scores[2]

                            log_file.write("precision for target {} : {}".format(target, precision))
                            log_file.write("recall for target {} : {}".format(target, recall))
                            log_file.write("fscore for target {} : {}".format(target, fscore))
                            log_file.flush()
                            np.save('./data/output/predictions/preds_{}_{}_{}_{}_{}_{}'.format(target, time_stamp, lambda_w, dropout, recurrent_dimension, patience), preds)
                            print("finished target: {}\n".format(target))

if __name__ == "__main__":
    main()
        