import sys
import os
import json
import numpy as np
import datetime
from scoring.metrics import Metrics
from scoring.metrics_objects import MetricGroundTruth, MetricWarning
from scoring.formatting_functions import format_gt, format_warn
from scoring.pair_objects_for_notebook import Pair
from munkres import Munkres


def main():
    gt_file = './ground-truth/armstrong-setcore-20171009.json'
    # warning_dir = './cause-effect/warnings/'
    warning_dir = '/Users/tozammel/cause/isi-code/effect-forecasting-models/warnings_jul_sep_17/armstrong_malicious-email/'

    start_date = datetime.date(2017, 7, 1)
    end_date = datetime.date(2017, 7, 31)
    target_org = 'armstrong'

    # Potentially filter by event type
    #   None -> Score for ALL event types
    #   Otherwise, restrict to 'endpoint-malware', 'malicious-email', or 'malicious-destination'
    event_type = None

    # First need to parse input data
    warnings = load_warnings(warning_dir, target_org, event_type, start_date,
                             end_date)
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
            if M.base_criteria(events[e_idx], warnings[w_idx]):
                # If so, calculate quality score
                pair = Pair.build(warnings[w_idx], events[e_idx],
                                  'fake-performer', 'fake-provider')
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

    avg_qs /= len(valid_pairings)
    avg_qs = -1 * avg_qs

    # calculate recall
    recall = nMatched / (nMatched + nUnmatchedGT)

    # calculate precision  (Srawls: Note: comment below is straight from official code, not mine)
    # should these include warns and gts from the previous month?
    precision = nMatched / (nMatched + nUnmatchedW)

    print("Precision = %0.2f%%" % (100 * precision))
    print("Recall = %0.2f%%" % (100 * recall))
    print("Average Quality Score = %0.2f" % avg_qs)

    return


def load_gt(gt_file, target_org, event_type, start_date, end_date):
    with open(gt_file, 'r', encoding='latin-1') as fh:
        gt_data = json.load(fh)

    events = []
    for event in gt_data:
        # Make sure event is of right target org
        if event['ground_truth']['target_organization'] != target_org:
            continue

        # Make sure event is in right date range
        event_date = datetime.datetime.strptime(
            event['ground_truth']['reported'], "%Y-%m-%dT%H:%M:%S.%fZ").date()
        if event_date < start_date or event_date > end_date:
            continue

        # Make sure event is of right event type, if filtering by event_type
        if not event_type is None:
            if event['ground_truth']['event_type'] != event_type:
                continue

        event_formatted = format_gt(event['ground_truth'])
        event_object = MetricGroundTruth.from_dict(event_formatted)
        events.append(event_object)

    return events


def load_warnings(warning_dir, target_org, event_type, start_date, end_date):
    warnings = []

    for f in os.listdir(warning_dir):
        if 'warning' in f:
            with open(os.path.join(warning_dir, f), 'r') as fh:
                warning = json.load(fh)

            if not 'occurred' in warning['warning']:
                # This is an old warning in different format, skip over
                continue


                # Make sure warning is of right target organization
            # if not 'target_organization' in warning['warning']:
            #    # For now we will just let all warnings w/o target_organization go through; these appear to be only endpoint-malware warnings
            #    continue

            if ('target_organization' in warning['warning']) and (
                        warning['warning'][
                            'target_organization'] != target_org):
                continue

            # Make sure warning is within right date range
            warning_date = datetime.datetime.strptime(
                warning['warning']['occurred'], "%Y-%m-%dT%H:%M:%SZ").date()
            if warning_date < start_date or warning_date > end_date:
                continue

            # Make sure warning is of right event type, if filtering by event_type
            if not event_type is None:
                if warning['event_type'] != event_type:
                    continue

            # Warning is good, let's load it in
            warning_formatted = format_warn(warning['warning'])
            warning_object = MetricWarning.from_dict(warning_formatted)

            # Need to manually add submitted time because submit_timestamp is in sepearte warning hierachy
            # Also need to add fake milisecond string to timestamp
            warning_object.submitted = warning['submit_timestamp'].replace(
                ":00Z", ":00.000000Z")
            warning_object.occurred = warning_object.occurred.replace(":00Z",
                                                                      ":00.000000Z")

            warnings.append(warning_object)

    return warnings


main()
