import sys
import os
import json
import numpy as np
import datetime
import pdb
import io
import csv
from scoring.metrics import Metrics
from scoring.metrics_objects import MetricGroundTruth, MetricWarning
from scoring.formatting_functions import format_gt, format_warn
from scoring.pair_objects_for_notebook import Pair
from munkres import Munkres
from argparse import ArgumentParser


def main(args):
    gt_file = './armstrong-jan2018.json'
    #gt_file = './armstrong-setcore-20171115.json'
    #gt_file = '/Users/ashokdeb/Desktop/test/effect-forecasting-models/deb/warning_ARIMA_p_7_d_1_q_7_armstrong_endpoint-malware_2017-06-30.json'
    # warning_dir = './cause-effect/warnings/'
    warning_dir = './score_warnings/'


    mth = args.mth
    external_signal_name = args.ext.split('.')[0]
    method = args.method
    event_type =args.evt

    if mth == "July":
        start_date = datetime.date(2017, 7, 1)
        end_date = datetime.date(2017, 7, 31)
    elif mth == "August":
        start_date = datetime.date(2017, 8, 1)
        end_date = datetime.date(2017, 8, 31)
    elif mth == "September":
        start_date = datetime.date(2017, 9, 1)
        end_date = datetime.date(2017, 9, 30)
    elif mth == "October":
        start_date = datetime.date(2017, 10, 1)
        end_date = datetime.date(2017, 10, 31)
    elif mth == "November":
        start_date = datetime.date(2017, 11, 1)
        end_date = datetime.date(2017, 11, 30)
    elif mth == "December":
        start_date = datetime.date(2017, 12, 1)
        end_date = datetime.date(2017, 12, 31)
    elif mth == "January":
        start_date = datetime.date(2018, 1, 1)
        end_date = datetime.date(2018, 1, 31)


    #need to adjust
    #start_date = datetime.date(2017, 7, 1)
    #end_date = datetime.date(2017, 7, 31)
    #start_date = datetime.date(2017, 8, 1)
    #end_date = datetime.date(2017, 8, 31)
    #start_date = datetime.date(2017, 9, 1)
    #end_date = datetime.date(2017, 9, 30)
    

    target_org = args.target
    #target_org = 'knox'

    # Potentially filter by event type
    #   None -> Score for ALL event types
    #   Otherwise, restrict to 'endpoint-malware', 'malicious-email', or 'malicious-destination'
    #event_type = None
    

    # First need to parse input data
    warnings = load_warnings(warning_dir, target_org, event_type, start_date, end_date, external_signal_name, method)
    events = load_gt(gt_file, target_org, event_type, start_date, end_date)

    print("# warnings = %d" % len(warnings))
    print("# events = %d" % len(events))

    # Now need to perform matching on warnings + events
    # To do this first have to score every possible warning-event pair
    # The official pair_objects.py is heavily dependent on python classes *not* provided to us by govt, so we recreate here
    M = Metrics()
    matching_matrix = np.zeros((len(events), len(warnings)))
    matching_dict = dict()
    for e_idx in range(len(events)):
        for w_idx in range(len(warnings)):
            # Check if we meet base criteria threshold
            if warnings[w_idx].event_type == "endpoint-malware":
                date_th = 0.875
            elif warnings[w_idx].event_type == "malicious-email":
                date_th = 1.375
            elif warnings[w_idx].event_type == "malicious-destination":
                date_th = 1.625
            else:
                raise Exception("Unknown event_type: %s" % warnings[w_idx].event_type)

            if M.base_criteria(events[e_idx], warnings[w_idx], thr2=date_th):
            	pair = Pair.build(warnings[w_idx], events[e_idx], 'fake-performer', 'fake-provider')
            	matching_matrix[e_idx, w_idx] = -pair.quality;
            	matching_dict["%d,%d" % (e_idx, w_idx)] = pair


    # Now do Hungarian matching
    munk = Munkres()
    pairings = munk.compute(matching_matrix.tolist())
    valid_pairings = list(
        filter(lambda p: matching_matrix[p[0], p[1]] != 0, pairings))

    nMatched = len(valid_pairings)
    nUnmatchedGT = len(events) - nMatched
    nUnmatchedW = len(warnings) - nMatched

    avg_qs = 0
    for e_idx, w_idx in valid_pairings:
        # pair = matching_dict["%d,%d" % (e_idx, w_idx)]
        # avg_qs += pair.quality
        avg_qs += matching_matrix[e_idx, w_idx]

    if len(valid_pairings) !=0:
        avg_qs /= len(valid_pairings)
        avg_qs = -1 * avg_qs
        recall = nMatched / (nMatched + nUnmatchedGT)
        precision = nMatched / (nMatched + nUnmatchedW)
        f1=(2*precision*recall)/(precision+recall)
    else:
        avg_qs=recall=precision=f1=0

    print("Precision = %0.2f%%" % (100 * precision))
    print("Recall = %0.2f%%" % (100 * recall))
    print("Average Quality Score = %0.2f" % avg_qs)

    new_results=[target_org, event_type, mth, len(events), len(warnings), external_signal_name,np.around(100 * precision, decimals=2) ,np.around(100*recall,decimals=2),np.around(100*f1,decimals=2),np.around(avg_qs,decimals=2)]

    with open('./output/results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(new_results)
    return


def load_gt(gt_file, target_org, event_type, start_date, end_date):
    with io.open(gt_file, 'r', encoding='latin-1') as fh:
        gt_data = json.load(fh)
    
    events = []
    for event in gt_data:
        # Make sure event is of right target org
        if event['ground_truth']['target_organization'] != target_org:
            continue

        # Make sure event is in right date range
        
        try:
            event_date= datetime.datetime.strptime(event['ground_truth']['reported'], "%Y-%m-%dT%H:%M:%S.%fZ").date()
        except:
            event_date= datetime.datetime.strptime(event['ground_truth']['reported'], "%Y-%m-%dT%H:%M:%SZ").date()    
        if event_date < start_date or event_date > end_date:
            continue

        # Make sure event is of right event type, if filtering by event_type
        if event['ground_truth']['event_type'] != event_type:
                continue

        event_formatted = format_gt(event['ground_truth'])
        event_object = MetricGroundTruth.from_dict(event_formatted)
        events.append(event_object)

    return events


def load_warnings(warning_dir, target_org, event_type, start_date, end_date,
                  external_signal_name, method):
    warnings2 = []

    for f in os.listdir(warning_dir):
        if external_signal_name not in f or method not in f:
            continue
        if 'warning' in f:
            print('Loading %s' % f)
            n1 = len(warnings2)
            print(n1)
            with open(os.path.join(warning_dir, f), 'r') as fh:
                warnings = json.load(fh)
                print('json loaded')
                print(len(warnings))
            for warning in warnings['warnings']:
                print(warning['warning']['occurred'])
                  
                # Make sure warning is within right date range
                warning_date = datetime.datetime.strptime(warning['warning']['occurred'], "%Y-%m-%dT%H:%M:%SZ").date()
                print(warning_date)
                print(start_date)
                print(end_date)
                if warning_date < start_date or warning_date > end_date:
                    continue

                # Make sure warning is of right event type, if filtering by event_type
                print(warning['warning']['event_type']) 
                if warning['warning']['event_type'] != event_type:
                    continue

                # Warning is good, let's load it in
                warning_formatted = format_warn(warning['warning'])
                warning_object = MetricWarning.from_dict(warning_formatted)

                # Need to manually add submitted time because submit_timestamp is in sepearte warning hierachy
                # Also need to add fake milisecond string to timestamp
                if 'submit_timestamp' in warning:
                    warning_object.submitted = warning['submit_timestamp'].replace(":00Z", ":00.000000Z")
                else:
                    warning_object.submitted = warning['created'].replace(":00Z", ":00.000000Z")

                warning_object.occurred = warning_object.occurred.replace(":00Z", ":00.000000Z")

                warnings2.append(warning_object)
            n2 = len(warnings2)
            print('# warnings in %s is %d' % (f, n2 - n1))

    return warnings2


if __name__ == "__main__":
    import sys

    parser=ArgumentParser()
    parser.add_argument('-ext', '--ext')
    parser.add_argument('-mth', '--mth')
    parser.add_argument('-method', '--method')
    parser.add_argument('-evt', '--evt')
    parser.add_argument('-target','--target')

    args = parser.parse_args()
    main(args)
