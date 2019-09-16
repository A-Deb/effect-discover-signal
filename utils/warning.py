#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


class Warning(object):
    '''
    Snippet to generate models following the specifications\
    provided by Lockheed Martin.
    '''
    def __init__(self):
        self.type = None
        self.id = None
        self.version = None
        self.confidence = None
        self.event_type = None
        self.submitted = None
        self.predicted_time = None
        self.targets = list()
        self.event_details = list()
        self.threat_actor = None
        self.ttps = list()
        self.linked_events = list()
        self.notes = None

    def add_target(self, name, identity_class, industry):
        new_target = dict(name=name, identity_class=identity_class,
                          industry=industry)
        self.targets.append(new_target)

    def add_event_details(self, event_detail_type, sensor_classification=None,
                          sensor=None, number_observed=1, first_observed=None):
        new_event = dict(type=event_detail_type,
                         sensor_classification=sensor_classification,
                         sensor=sensor, number_observed=number_observed)
        self.event_details.append(new_event)

    def add_predicted_time(self, timestamp, min_timestamp=None,
                           max_timestamp=None):
        pred_time = dict(timestamp=timestamp, min=min_timestamp,
                         max=max_timestamp)
        self.predicted_time = pred_time

    def show(self):
        print('Type: ', self.type)
        print('Id: ', str(self.id))
        print('Category: ', self.event_type)
        print('Created: ', self.submitted)
        print('Confidence: ', str(self.confidence))
        print('Victim Target(s):')

        for target in self.targets:
            print('\tName: ', target['name'])
            print('\tIdentity Class: ', target['identity_class'])
            print('\tSector(s): ', target['industry'])

        if len(self.targets) == 0:
            print('\t> No victims!')
            print('\t> Use add_target() to add a new victim to this model.')

    def save(self, filepath):
        import json
        # path_to_file = 'warning_%s.json' % (self.id)

        with open(filepath, 'w') as f:
            json.dump(self, f, default=lambda o: o.__dict__,
                      sort_keys=True, indent=4)
        print("Warning is saved to %s" % filepath)
        # print('Model %s saved to %s' % (self.id, path_to_file))


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--text', default='Hello World!',
                        help='Help string')
    options = parser.parse_args()
    print(options.text)

    '''
    Example of minimal warning
    '''

    # initialize a new object of class Warning
    warning_obj = Warning()

    # add single attributes
    warning_obj.type = 'incident'
    warning_obj.id = '4'
    warning_obj.category = 'malicious-email'
    warning_obj.created = '2016-04-06T20:03:48Z'
    warning_obj.confidence = '0.8'
    # warning_obj.first_observed['prediction'] = '2016-04-06T13:01:36Z'
    # warning_obj.first_observed['stdev'] = '86400'

    # add victims
    # warning_obj.add_victim('Dexter', 'organization', ['financial-services'])
    # warning_obj.add_victim('Tesla Motors', 'organization',
    #                        ['energy-storage', 'electric-cars'])

    # print attributes using method show()
    warning_obj.show()

    # save to JSON using method save()
    warning_obj.save()

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))