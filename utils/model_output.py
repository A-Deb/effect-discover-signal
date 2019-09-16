#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from utils.warning_version_2 import generate_warning_objects
from utils.warning_version_2 import generate_warning_objects_knox

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


class ModelOutput(object):
    """
    Snippet to generate models following the specifications\
    provided by Lockheed Martin.
    """

    def __init__(self):
        # header
        self.warnings = None
        # self.output_name = None
        # self.output_type = None
        # self.audit_trail_id = None
        # self.processed_by = None
        # self.created = None
        # self.warning_fields = None

    def show(self):
        print(self.warnings)
        # print('Output Name:', self.output_name)
        # print('Output Type:', self.output_type)
        # print('Audit Trail Id:', self.audit_trail_id)
        # print('Processed By: ', self.processed_by)
        # print('Created: ', self.created)

    def save(self, filepath):
        import json
        # path_to_file = 'warning_%s.json' % (self.id)

        with open(filepath, 'w') as f:
            json.dump(self, f, default=lambda o: o.__dict__,
                      sort_keys=True, indent=4)
        print("Warning is saved to %s" % filepath)
        # print('Model %s saved to %s' % (self.id, path_to_file))


def write_model_output(options, ts_pred, model_id, output_name):
    import os
    modelout_obj = ModelOutput()

    current_time = pd.datetime.now()
    warn_date = ts_pred.index[0] - pd.Timedelta(days=1)
    if warn_date < current_time:
        warn_timestamp = warn_date + pd.Timedelta(
            hours=current_time.hour, minutes=current_time.minute,
            seconds=current_time.second, microseconds=current_time.microsecond)
    else:
        warn_timestamp = current_time

    # header
    modelout_obj.warnings = []

    for time, count in ts_pred.iteritems():
        warn_obj = dict()
        warn_obj["output_name"] = output_name
        warn_obj["output_type"] = "candidate_warning"
        warn_obj["audit_trail_id"] = None
        warn_obj["processed_by"] = model_id
        warn_obj["created"] = warn_timestamp.isoformat() + "Z"

        # payload
        warn_body = dict()
        warn_body["type"] = "warning"
        # warn_body["id"] = model_id + "_" + warn_date.strftime("%Y%m%d")
        # warn_body["id"] = model_id + "_" + warn_timestamp.strftime("%Y%m%d_%H%M")
        warn_body["id"] = model_id + "_" + warn_timestamp.strftime("%Y%m%d")
        warn_body["version"] = "1"
        warn_body["confidence"] = 1.0
        warn_body["event_type"] = options.event_type
        warn_body["submitted"] = warn_timestamp.isoformat() + "Z"

        # make the time midday
        time = time + pd.Timedelta(hours=12)
        pred_time = dict(timestamp=time.isoformat() + "Z",
                         min=None, max=None)
        warn_body["predicted_time"] = pred_time

        warn_body["targets"] = []
        new_target = dict(name="dexter", identity_class="organization",
                          industry="financial-services")
        warn_body["targets"].append(new_target)

        warn_body["event_details"] = []
        new_event = dict(type=options.event_type_broad,
                         sensor_classification=options.data_source,
                         sensor=options.sensor,
                         number_observed=round(count))
        warn_body["event_details"].append(new_event)
        warn_obj["warning"] = warn_body
        modelout_obj.warnings.append(warn_obj)

    # save the warning
    # filename = "warning_" + warn_body["event_type"] + "_" + warn_body["id"] + \
    #            ".json"
    filename = "warning_" + warn_body["id"] + ".json"
    filepath = os.path.join(options.warning_dir, options.data_source, filename)
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    # print("Writing warning to the file:", filepath)
    # modelout_obj.show()
    modelout_obj.save(filepath)


def get_target_orgnization(options):
    target_org = None
    if options.data_source == 'ransomware_locky':
        target_org = "abuse.ch"
    elif options.data_source == 'ransomware_cerber':
        target_org = "abuse.ch"
    elif options.data_source == 'dexter_endpoint-malware':
        target_org = "dexter"
    elif options.data_source == 'dexter_malicious-email':
        target_org = "dexter"
    elif options.data_source == 'dexter_malicious-destination':
        target_org = "dexter"
    elif options.data_source == 'armstrong_endpoint-malware':
        target_org = "armstrong"
    elif options.data_source == 'armstrong_malicious-destination':
        target_org = "armstrong"
    elif options.data_source == 'armstrong_malicious-email':
        target_org = "armstrong"
    elif options.data_source == 'knox_endpoint-malware':
        target_org = "knox"
    elif options.data_source == 'knox_malicious-destination':
        target_org = "knox"
    elif options.data_source == 'knox_malicious-email':
        target_org = "knox"
    else:
        target_org = None

    return target_org


def write_model_output_v2(options, ts_pred, model_id, output_name,
                          save_warning=True):
    modelout_obj = ModelOutput()

    # current_time = pd.datetime.now()
    # warn_date = ts_pred.index[0] - pd.Timedelta(days=1)

    # if warn_date < current_time:
    #     warn_timestamp = warn_date + pd.Timedelta(
    #         hours=current_time.hour, minutes=current_time.minute,
    #         seconds=current_time.second, microseconds=current_time.microsecond)
    # else:
    #     warn_timestamp = current_time

    warn_timestamp = options.warn_start_date - pd.Timedelta(days=1)

    # header
    modelout_obj.warnings = []
    warn_idx = 0
    for time, count in ts_pred.iteritems():
        if count > 0:
            warn_obj = dict()
            # if not hourly_model:
            #     time = time + pd.Timedelta(hours=12)  # Make the time midday
            warn_obj = generate_warning_objects(warn_obj,
                                                get_target_orgnization(options),
                                                options.event_type, count,
                                                warn_idx,
                                                output_name, model_id,
                                                warn_timestamp,
                                                # warn_timestamp.isoformat() + "Z",
                                                time.isoformat() + "Z")
            warn_idx += count
            # print(len(warn_obj))
            # print(warn_obj)
            # print("")
            modelout_obj.warnings.extend(warn_obj)

    if save_warning:
        # save the warning
        # print(warn_obj)
        # print(options.event_type)
        # print(model_id)
        # id_prefix = warn_obj[0]['warning']['id'].rsplit("T", 1)[0]
        id_prefix = modelout_obj.warnings[0]['warning']['id'].rsplit("T", 1)[0]
        # print(id_prefix)
        filename = "warning_" + id_prefix + ".json"
        filepath = os.path.join(options.warning_dir, options.data_source,
                                filename)
        os.makedirs(os.path.split(filepath)[0], exist_ok=True)
        print("Writing warning to the file:", filepath)
        # modelout_obj.show()
        modelout_obj.save(filepath)
    return modelout_obj


def write_model_output_v3(options, ts_pred, model_id, output_name,
                          save_warning=True):
    modelout_obj = ModelOutput()
    warn_timestamp = options.warn_start_date - pd.Timedelta(days=1)

    if options.feature_dist_file is None:
        if options.event_type == "malicious-email":
            feature_dist_file = "data/knox/knox-distribution_malicious-email.json"
        elif options.event_type == "malicious-destination":
            feature_dist_file = "data/knox/knox-distribution_malicious-destination.json"
        elif options.event_type == "endpoint-malware":
            feature_dist_file = "data/knox/knox-distribution_endpoint-malware.json"
        else:
            print("A valid feature distribution file is required!")
    else:
        feature_dist_file = options.feature_dist_file
    # header
    modelout_obj.warnings = []
    warn_idx = 0
    for time, count in ts_pred.iteritems():
        if count > 0:
            warn_obj = dict()
            # if not hourly_model:
            #     time = time + pd.Timedelta(hours=12)  # Make the time midday
            warn_obj = generate_warning_objects_knox(
                warn_obj, get_target_orgnization(options),
                options.event_type, count, warn_idx, output_name, model_id,
                warn_timestamp, time.isoformat() + "Z", feature_dist_file)

            warn_idx += count
            # print(len(warn_obj))
            # print(warn_obj)
            # print("")
            modelout_obj.warnings.extend(warn_obj)

    # save the warning
    # print(warn_obj)
    # print(options.event_type)
    # print(model_id)

    if save_warning:
        # id_prefix = warn_obj[0]['warning']['id'].rsplit("T", 1)[0]
        id_prefix = modelout_obj.warnings[0]['warning']['id'].rsplit("T", 1)[0]
        # print(id_prefix)
        filename = "warning_" + id_prefix + ".json"
        filepath = os.path.join(options.warning_dir, options.data_source,
                                filename)
        os.makedirs(os.path.split(filepath)[0], exist_ok=True)
        print("Writing warning to the file:", filepath)
        # modelout_obj.show()
        modelout_obj.save(filepath)
    return modelout_obj


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--text', default='Hello World!',
                        help='Help string')
    parser.add_argument(
        '-d', '--data-source', default='knox_malicious-email')
    parser.add_argument(
        '--event-type', default='malicious-email')
    parser.add_argument(
        '--feature-dist-file', default=None)
    parser.add_argument(
        '--warn-start-date',
        default=pd.datetime.now().date() + pd.Timedelta(days=1))
    parser.add_argument(
        '--warning-dir',
        default='warnings',
        help='Save json warning into this directory.')
    options = parser.parse_args()
    print(options.text)

    """
    test knox warnings
    """
    warn_date = pd.to_datetime("2017-09-25").date()
    ts_pred = pd.Series([1], index=[warn_date])
    model_id = "DaywiseBaserate" + "_" + options.data_source
    write_model_output_v3(options, ts_pred, model_id,
                          output_name="daywise_baserate_warning")

    '''
    Example of minimal warning
    '''

    # initialize a new object of class Warning
    # warning_obj = Warning()

    # add single attributes
    # warning_obj.type = 'incident'
    # warning_obj.id = '4'
    # warning_obj.category = 'malicious-email'
    # warning_obj.created = '2016-04-06T20:03:48Z'
    # warning_obj.confidence = '0.8'
    # warning_obj.first_observed['prediction'] = '2016-04-06T13:01:36Z'
    # warning_obj.first_observed['stdev'] = '86400'

    # add victims
    # warning_obj.add_victim('Dexter', 'organization', ['financial-services'])
    # warning_obj.add_victim('Tesla Motors', 'organization',
    #                        ['energy-storage', 'electric-cars'])

    # print attributes using method show()
    # warning_obj.show()

    # save to JSON using method save()
    # warning_obj.save()


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
