#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json


class modelWrapper():
    '''
    Snippet to generate models following the specifications\
    provided by Lockheed Martin.
    '''
    def __init__(self):
        self.id = None
        self.model_name = None
        self.model_version = None
        self.model_type = None
        self.repository = None
        self.author = None
        self.model_input = None
        self.parameters = list()
        self.model_description = None
        self.templated_narrative_hypothesis = None
        self.templated_narrative_context = None

    def show(self):
        print('Id: ', self.id)
        print('Name: ', self.model_name)
        print('Version: ', self.model_version)
        print('Type: ', self.model_type)
        print('Repository: ', self.repository)
        print('Author: ', self.author)
        print('Model Inputs: ', self.model_input)
        print('Model Parameters: ')
        for param in self.parameters:
            print(param)

        print()
        print('Model Description:')
        print(self.model_description)
        print()
        print('Templated Narrative Hypothesis:')
        print(self.templated_narrative_hypothesis)
        print()
        print('Templated Narrative Context:')
        print(self.templated_narrative_context)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self, f, default=lambda o: o.__dict__,
                      sort_keys=True, indent=4)
        print('Model metadata is saved to %s' % (filepath))


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--text', default='Hello World!',
                        help='Help string')
    options = parser.parse_args()
    print(options.text)

    '''
    Usage of modelWrapper class
    '''

    # initialize a new object of class modelWrapper
    HMM = modelWrapper()

    # add single attributes
    ##########################################################################
    HMM.id = 'hmm_1.0'  # new id when model changes version
    HMM.model_name = 'time-series-model'
    HMM.model_version = 'v0.0.1'
    HMM.model_type = 'forecast_model'
    HMM.repository = 'https://github.com/usc-isi-i2/effect-forecasting-models/tree/master/model/hmm'
    HMM.author = 'ISI USC'
    HMM.model_inputs = '...'  # static part of ES queries

    HMM.parameters.append(('name', 1))
    # add parameters in the form of name-value pairs (tuple)

    HMM.model_description = 'processes time series data'
    HMM.templated_narrative_hypothesis = '''write here the narrative hypothesis
    multiline
    '''
    HMM.templated_narrative_context = '''write here the narrative context
    multiline
    '''
    ###########################################################################

    # print attributes using method show()
    HMM.show()

    # save to JSON using method save()
    HMM.save()

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))