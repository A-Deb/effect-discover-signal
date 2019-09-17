"""Pair object module to use in the demonstration notebook."""
# pylint: disable=deprecated-lambda,invalid-name,no-self-use,protected-access,too-few-public-methods
import json
from .metrics_objects import MetricGroundTruth, MetricWarning

from .metrics import Metrics

TIME_PATTERN = "%Y-%m-%dT%H:%M:%S.%fZ"

METRIC_VERSION = "2.0"


def attrgetter(attr):
    """Return a getter function for the specified attribute."""
    def get_attr_func(self):
        """Return inner function."""
        return getattr(self, attr)
    return get_attr_func


def attrdeleter(attr):
    """Return a deleter function for the specified attribute that sets an attribute to None."""
    def del_attr_func(self):
        """Return inner function."""
        if hasattr(self, attr):
            setattr(self, attr, None)
    return del_attr_func


class Pair(object):  # pylint: disable=too-many-instance-attributes
    """Pair consisting of ground truth and warning."""

    def __init__(self):
        """Set default values for the Pair class."""
        self._warning = None
        self._ground_truth = None
        self._matched = None
        self._metric_version = None
        self._performer = None
        self._provider = None
        self._lead_time = None
        self._utility_time = None
        self._confidence = None
        self._probability = None
        self._quality = None
        self._event_type_similarity = None
        self._event_details_similarity = None
        self._occurrence_time_similarity = None
        self._targets_similarity = None
        self._metrics = Metrics()

    def csv_row(self):
        """Generate human readable string representation of the pair.

        :return: a human readable string representation of the object
        :rtype: str
        """
        return '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' % (int(self.warning.id),
                                                                           int(self.ground_truth.id),
                                                                           self.matched,
                                                                           self.metric_version,
                                                                           self.performer,
                                                                           self.provider,
                                                                           self.lead_time,
                                                                           self.utility_time,
                                                                           self.probability,
                                                                           self.quality,
                                                                           self.event_type_similarity,
                                                                           self.event_details_similarity,
                                                                           self.occurrence_time_similarity,
                                                                           self.targets_similarity)

    def to_json_dict(self):
        """Convert object to JSON dictionary."""
        attributes = filter(lambda x: x[0] not in ('_ground_truth', '_warnings', '_metrics'), self.__dict__.items())
        json_content = ', '.join(['\"%s\": %s' % (x[0][1:], x[1]) for x in attributes])
        json_content = '{' + json_content + '}'
        return json.dumps(json_content)

    def csv_head(self):
        """Generate header row for CSV."""
        return ('warning_id, ground_truth_id, matched, metric_version, performer, provider, lead_time, utility_time, '
                'probability, quality, event_type_similarity, event_details_similarity, occurrence_time_similarity, '
                'targets_similarity')

    warning = property(fget=attrgetter("_warning"), fdel=attrdeleter("_warning"))

    @warning.setter
    def warning(self, value):
        """Set warning component.

        :param value: warning component of the pairing
        :type value: MetricWarning
        """
        self._warning = value

    ground_truth = property(fget=attrgetter("_ground_truth"), fdel=attrdeleter("_ground_truth"))

    @ground_truth.setter
    def ground_truth(self, value):
        """Set ground truth component.

        :param value: ground truth component of the pairing
        :type value: MetricGroundTruth
        """
        self._ground_truth = value

    matched = property(fget=attrgetter("_matched"), fdel=attrdeleter("_matched"))

    @matched.setter
    def matched(self, value):
        """Set matched."""
        self._metric_version = value

    metric_version = property(fget=attrgetter("_metric_version"), fdel=attrdeleter("_metric_version"))

    @metric_version.setter
    def metric_version(self, value):
        """Set metric version."""
        self._metric_version = value

    performer = property(fget=attrgetter("_performer"), fdel=attrdeleter("_performer"))

    @performer.setter
    def performer(self, value):
        """Set performer."""
        self._performer = value

    provider = property(fget=attrgetter("_provider"), fdel=attrdeleter("_provider"))

    @provider.setter
    def provider(self, value):
        """Set provider."""
        self._provider = value

    lead_time = property(fget=attrgetter("_lead_time"), fdel=attrdeleter("_lead_time"))

    @lead_time.setter
    def lead_time(self, value):
        """Set lead time."""
        self._lead_time = value

    utility_time = property(fget=attrgetter("_utility_time"), fdel=attrdeleter("_utility_time"))

    @utility_time.setter
    def utility_time(self, value):
        """Set utility time."""
        self._utility_time = value

    confidence = property(fget=attrgetter("_confidence"), fdel=attrdeleter("_confidence"))

    @confidence.setter
    def confidence(self, value):
        """Set confidence."""
        self._confidence = value

    probability = property(fget=attrgetter("_probability"), fdel=attrdeleter("_probability"))

    @probability.setter
    def probability(self, value):
        """Set probability."""
        self._probability = value

    quality = property(fget=attrgetter("_quality"), fdel=attrdeleter("_quality"))

    @quality.setter
    def quality(self, value):
        """Set quality."""
        self._quality = value

    event_type_similarity = property(fget=attrgetter("_event_type_similarity"),
                                     fdel=attrdeleter("_event_type_similarity"))

    @event_type_similarity.setter
    def event_type_similarity(self, value):
        """Set event type similarity."""
        self._event_type_similarity = value

    event_details_similarity = property(fget=attrgetter("_event_details_similarity"),
                                        fdel=attrdeleter("_event_details_similarity"))

    @event_details_similarity.setter
    def event_details_similarity(self, value):
        """Set event details similarity."""
        self._event_details_similarity = value

    occurrence_time_similarity = property(fget=attrgetter("_occurrence_time_similarity"),
                                          fdel=attrdeleter("_occurrence_time_similarity"))

    @occurrence_time_similarity.setter
    def occurrence_time_similarity(self, value):
        """Set occurrence time similarity."""
        self._occurrence_time_similarity = value

    targets_similarity = property(fget=attrgetter("_targets_similarity"), fdel=attrdeleter("_targets_similarity"))

    @targets_similarity.setter
    def targets_similarity(self, value):
        """Set targets similarity."""
        self._targets_similarity = value

    @classmethod
    def build(cls, warn, gt, performer, provider):
        """Build pair using warning and ground truth from performer and provider (respectively).

        :param warn: warning json dict to be compared to gt
        :type warn: dict
        :param gt: ground truth json dict to be compared to warn
        :type gt: dict
        """
        #mgt = MetricGroundTruth()
        #mwn = MetricWarning()

        #warn = mwn.from_dict(warn)
        #gt = mgt.from_dict(gt)

        pair = cls()

        pair.warning = warn
        pair.ground_truth = gt
        pair.metric_version = METRIC_VERSION
        pair.performer = performer
        pair.provider = provider

        pair.lead_time = pair._metrics.lead_time_score(gt, warn)
        pair.utility_time = pair._metrics.utility_time_score(gt, warn)
        pair.quality = pair._metrics.quality_score(gt, warn)
        pair.event_type_similarity = pair._metrics.event_type_score(gt, warn)
        pair.event_details_similarity = pair._metrics.event_details_score_all(gt, warn)
        pair.occurrence_time_similarity = pair._metrics.occ_time_score(gt, warn)
        pair.targets_similarity = pair._metrics.target_score(gt, warn)

        return pair

    def get_pair_ids(self):
        """Return the ids of warning and ground truth in a tuple."""
        return (self._warning.id, self._ground_truth.id)

    def pair_id_string(self):
        """Return a string representation of the tuple returned by get_pair_ids."""
        return '(' + str(self._warning.id) + ',' + str(self._ground_truth.id) + ')'

    def set_probability(self, m):
        """Set confidence of match.

        :param m: either 1 if matched or 0 if unmatched
        :type m: int
        """
        self._probability = self._metrics.probability_score(m, self._warning)
        self._matched = bool(m)


class PairBank(object):
    """Bank of pairs."""

    def __init__(self):
        """Initialize bank."""
        self._bank = []

    def __iter__(self):
        """Iterate over bank."""
        for b in self._bank:
            yield b

    def __getitem__(self, ii):
        """Get pair from bank."""
        return self._bank[ii]

    def __str__(self):
        """Return a csv row for each Pair in PairBank."""
        return '\n'.join([x.csv_row() for x in self])

    def add_pair(self, pair):
        """Add a Pair object."""
        self._bank.append(pair)

    def __add__(self, x):
        """Add one PairBank to another."""
        y = PairBank()
        y._bank = self._bank + x._bank
        return y
