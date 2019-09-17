import json
from datetime import datetime, timedelta
from munkres import Munkres
from metrics.metrics_objects import MetricGroundTruth, MetricWarning
from setcore.setcore_core.psql_connector import PSQLConnector, PSQLConnectionConfig
from metrics.metrics_schema import PairBankDB
from metrics.settings import DB_CONFIG
from metrics.pair_resource import PairDBResource
from metrics.bank_objects import Bank, GroundTruthBank, WarningBank

from metrics.metrics import Metrics

TIME_PATTERN = "%Y-%m-%dT%H:%M:%S.%fZ"

METRIC_VERSION = "2.0"


def unique(ll):
    coll = []
    for l in ll:
        if l not in coll:
            coll += [l]
    return coll


def attrgetter(attr):
    """
    Returns a getter function for the specified attribute
    """
    def get_attr_func(self):
        """returned inner function"""
        return getattr(self, attr)
    return get_attr_func


def attrdeleter(attr):
    """
    Returns a deleter function for the specified attribute that sets an attribute to None
    """
    def del_attr_func(self):
        """returned inner function"""
        if hasattr(self, attr):
            setattr(self, attr, None)
    return del_attr_func


class Pair(object):

    def __init__(self):
        """
        Sets default values for the Pair class
        """
        self._warning = None
        self._ground_truth = None
        self._matched = False
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
        """
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
        attributes = filter(lambda x: x[0] not in ('_ground_truth', '_warnings', '_metrics'), self.__dict__.items())
        json_content = ', '.join(['\"%s\": %s' % (x[0][1:], x[1]) for x in attributes])
        json_content = '{' + json_content + '}'
        return json.dumps(json_content)

    def db_row(self):
        return (self.metric_version,
                self.warning.id,
                self.ground_truth_id,
                self.lead_time,
                self.utility_time,
                self.quality,
                self.event_type_similarity,
                self.event_details_similarity,
                self.occurrence_time_similarity,
                self.targets_similarity,
                self.probability)

    def csv_head(self):
        return 'warning_id, ground_truth_id, matched, metric_version, performer, provider, lead_time, utility_time, probability, quality, event_type_similarity, event_details_similarity, occurrence_time_similarity, targets_similarity'

    warning = property(fget=attrgetter("_warning"), fdel=attrdeleter("_warning"))

    @warning.setter
    def warning(self, value):
        """
        :param value: warning component of the pairing
        :type value: MetricWarning
        """
        self._warning = value

    ground_truth = property(fget=attrgetter("_ground_truth"), fdel=attrdeleter("_ground_truth"))

    @ground_truth.setter
    def ground_truth(self, value):
        """
        :param value: ground truth component of the pairing
        :type value: MetricGroundTruth
        """
        self._ground_truth = value

    matched = property(fget=attrgetter("_matched"), fdel=attrdeleter("_matched"))

    @matched.setter
    def matched(self, value):

        self._metric_version = value

    metric_version = property(fget=attrgetter("_metric_version"), fdel=attrdeleter("_metric_version"))

    @metric_version.setter
    def metric_version(self, value):

        self._metric_version = value

    performer = property(fget=attrgetter("_performer"), fdel=attrdeleter("_performer"))

    @performer.setter
    def performer(self, value):

        self._performer = value

    provider = property(fget=attrgetter("_provider"), fdel=attrdeleter("_provider"))

    @provider.setter
    def provider(self, value):

        self._provider = value

    lead_time = property(fget=attrgetter("_lead_time"), fdel=attrdeleter("_lead_time"))

    @lead_time.setter
    def lead_time(self, value):

        self._lead_time = value

    utility_time = property(fget=attrgetter("_utility_time"), fdel=attrdeleter("_utility_time"))

    @utility_time.setter
    def utility_time(self, value):

        self._utility_time = value

    confidence = property(fget=attrgetter("_confidence"), fdel=attrdeleter("_confidence"))

    @confidence.setter
    def confidence(self, value):

        self._confidence = value

    probability = property(fget=attrgetter("_probability"), fdel=attrdeleter("_probability"))

    @probability.setter
    def probability(self, value):

        self._probability = value

    quality = property(fget=attrgetter("_quality"), fdel=attrdeleter("_quality"))

    @quality.setter
    def quality(self, value):

        self._quality = value

    event_type_similarity = property(fget=attrgetter("_event_type_similarity"), fdel=attrdeleter("_event_type_similarity"))

    @event_type_similarity.setter
    def event_type_similarity(self, value):

        self._event_type_similarity = value

    event_details_similarity = property(fget=attrgetter("_event_details_similarity"), fdel=attrdeleter("_event_details_similarity"))

    @event_details_similarity.setter
    def event_details_similarity(self, value):

        self._event_details_similarity = value

    occurrence_time_similarity = property(fget=attrgetter("_occurrence_time_similarity"), fdel=attrdeleter("_occurrence_time_similarity"))

    @occurrence_time_similarity.setter
    def occurrence_time_similarity(self, value):

        self._occurrence_time_similarity = value

    targets_similarity = property(fget=attrgetter("_targets_similarity"), fdel=attrdeleter("_targets_similarity"))

    @targets_similarity.setter
    def targets_similarity(self, value):

        self._targets_similarity = value

    @classmethod
    def build(cls, warn, gt, performer, provider):
        '''
        :param warn: warning json dict to be compared to gt
        :type warn: dict
        :param gt: ground truth json dict to be compared to warn
        :type gt: dict
        :param performer: name of performer
        :type performer: str
        :param provider: name of provider
        :type provider: str
        '''
        mgt = MetricGroundTruth()
        mwn = MetricWarning()

        warn = mwn.from_dict(warn)
        gt = mgt.from_dict(gt)

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

    @classmethod
    def build_unpaired_warn(cls, warn, performer, provider):
        '''
        :param warn: MetricWarning object representation of unpaired warning
        :type warn: MetricWarning
        :param performer: name of performer
        :type performer: str
        :param provider: name of provider
        :type provider: str
        '''

        pair = cls()

        pair.warning = warn
        pair.metrics_version = METRIC_VERSION
        pair.performer = performer
        pair.provider = provider
        pair.set_probability(0)

        return pair

    @classmethod
    def build_unpaired_gt(cls, gt, performer, provider):
        '''
        :param warn: MetricGroundTruth object representation of unpaired ground truth
        :type warn: MetricGroundTruth
        :param performer: name of performer
        :type performer: str
        :param provider: name of provider
        :type provider: str
        '''

        pair = cls()

        pair.ground_truth = gt
        pair.metrics_version = METRIC_VERSION
        pair.performer = performer
        pair.provider = provider

        return pair

    def get_pair_ids(self):
        '''
        returns the ids of warning and ground truth in a tuple
        '''
        return (self._warning.id, self._ground_truth.id)

    def pair_id_string(self):
        '''
        returns a string representation of the tuple returned by get_pair_ids
        '''
        return '('+str(self._warning.id)+','+str(self._ground_truth.id)+')'

    def sim_score(self):
        '''
        returns the mean similarity score for creating the cost matrix for pairing
        '''
        return self._mean_similarity

    def mean_pair_score(self):
        '''
        returns the mean of all collected scores
        '''
        scores = filter(lambda x: x is not None, [self._lead_time, self._utility_time, self._probability, (self._quality/4.0), self._mean_similarity])
        mean_score = self._metrics.mean(scores)
        return(mean_score)

    def set_probability(self, m):
        '''
        :param m: either 1 if matched or 0 if unmatched
        :type m: int
        '''
        self._matched = bool(m)
        self._probability = self._metrics.probability_score(m, self._warning)

    @classmethod
    def from_db_dict(cls, db_dict):
        '''
        :param db_dict: the dict that is generated by the sql commands
        :type db_dict: dict
        '''
        pair = cls()

        pair.metric_version = db_dict['metric_version']
        mgt = MetricGroundTruth()
        mwn = MetricWarning()
        pair.warning = mwn.from_dict({'id': db_dict['gt_id']})
        pair.ground_truth = mgt.from_dict({'id': db_dict['warn_id']})

        pair.lead_time = db_dict['lead_time']
        pair.utility_time = db_dict['utility_time']
        pair.quality = db_dict['quality']
        pair.event_type_similarity = db_dict['event_type_similarity']
        pair.event_details_similarity = db_dict['event_details_similarity']
        pair.occurrence_time_similarity = db_dict['occurrence_time_similarity']
        pair.targets_similarity = db_dict['targets_similarity']
        pair.probability_score = db_dict['probability']
        return pair


class PairBank(object):

    def __init__(self):
        super(self.__class__, self).init()
        self._metrics = Metrics()
        self._ground_truth_bank = None
        self._warning_bank = None
        self._from_db = None

    def __iter__(self):
        for b in self._bank:
            yield b

    def trim(self, list_of_pair_ids):
        '''
        returns a trimmed PairBank of Pair objects
        :param list_of_pair_ids: a list of warn/gt id tuples
        :type list_of_pair_ids: [(int, int)]
        '''

        trimmed = filter(lambda x: x.get_pair_ids() in list_of_pair_ids, self)
        pb_trim = PairBank()
        pb_trim._bank = trimmed
        return pb_trim

    def __getitem__(self, ii):
        '''
        :param ii: the ii_th item in the PairBank
        :type ii: int
        '''
        return self._bank[ii]

    def __len__(self):
        '''
        returns length of pair bank list
        '''
        return len(self._bank)

    def get_by_ids(self, i, j):
        '''
        :param i: warning id
        :type i: int
        :param j: ground truth id
        :type j: int
        '''
        warn = self.get_from_warns_by_id(i).to_dict()
        gt = self.get_from_gt_by_id(j).to_dict()
        return Pair.build(warn, gt, self._warning_bank.performer, self._ground_truth_bank.provider)

    def get_from_warns_by_id(self, i):
        '''
        :param i: warning id
        :type i: int
        '''
        return filter(lambda x: x.id == i, self._warning_bank)[0]

    def get_from_gt_by_id(self, j):
        '''
        :param j: ground truth id
        :type j: int
        '''
        return filter(lambda x: x.id == j, self._ground_truth_bank)[0]

    def get_by_ids_from_banks(self, i, j):
        '''
        :param i: warning id
        :type i: int
        :param j: ground truth id
        :type j: int
        '''
        # don't make pair if in _from_db dict
        if (i, j) in [x.get_pair_ids() for x in self._from_db]:
            pair = filter(lambda x: x.get_pair_ids() == (i, j), self._from_db)[0]
            return pair
        else:
            warn = self.get_from_warns_by_id(i)
            gt = self.get_from_gt_by_id(j)
            pair = Pair()
            mkpair = pair.build(warn, gt, self.performer, self.provider)
            self.add_pair(mkpair)
            return mkpair

    def __str__(self):
        '''
        returns a csv row for each Pair in PairBank
        '''
        return '\n'.join([x.csv_row() for x in self])

    def add_pair(self, pair):
        '''
        Add a Pair object
        '''
        self._bank.append(pair)

    def __add__(self, x):
        '''
        An addition operation for adding one PairBank to another
        '''
        y = PairBank()
        y._bank = self._bank + x._bank
        return y

    def get_warn_ids(self):
        '''
        returns the warning ids of the pair bank
        '''
        if self._warnging_bank is not None:
            ids = unique([x.id for x in self._warning_bank])
        else:
            ids = None
        return ids

    def get_gt_ids(self):
        '''
        returns ground truth ids of the pair bank
        '''
        if self._ground_truth_bank is not None:
            ids = unique([x.id for x in self._ground_truth_bank])
        else:
            ids = None
        return ids

    def base_crit(self, i, j):
        '''
        :param i: warning id
        :type i: int
        :param j: ground truth id
        :type j: int
        '''
        wn = self.get_from_warns_by_id(i)
        gt = self.get_from_gt_by_id(j)
        return self._metrics.base_criteria(gt, wn)

    def run_hungarian(self):
        '''
        Run the hungarian algorithm to get valid Pair objects
        '''
        warn_ids = self.get_warn_ids()
        gt_ids = self.get_gt_ids()

        quals = [[-self.get_by_ids_from_banks(i, j).quality if self.base_crit(i, j) else 0.0 for j in warn_ids] for i in gt_ids]

        munk = Munkres()
        pairings = munk.compute(quals)
        pairings_ids = [(warn_ids[x], gt_ids[y]) for x, y in pairings]
        unpaired_warns_ids = filter(lambda x: x not in [y[0] for y in pairings_ids], warn_ids)
        unpaired_gt_ids = filter(lambda x: x not in [y[1] for y in pairings_ids], gt_ids)

        return pairings_ids, unpaired_warns_ids, unpaired_gt_ids

    def generate_pairs(self, performer, provider, start_date, end_date):
        '''
        This method generates the ground truth/warning pairs and unpaired ground
        and warnings by building the warning and ground truth banks with in the
        pair bank, pulling the already existing pairs from the persistent pair
        bank to check the boundaries, and calculates recall and precision.
        :param performer: name of performer
        :type performer: str
        :param provider: name of provider
        :type provider: str
        :param start_date: start of the scoring period
        :type start_date: str in the pattern of YYYY-MM-DD
        :param end_date: end of the scoring period
        :type end_date: str in the pattern of YYYY-MM-DD
        :return: a list of warn/gt pairs, a list of unpaired warns and gt, a recall score, and a precision score
        :rtype: ([Pair], [Pair], float, float)
        '''

        # fill WarningBank
        wb = WarningBank(performer, start_date, end_date)
        wb.fill()
        self._warn_bank = wb

        # fill GroundTruthBank
        gtb = GroundTruthBank(provider, start_date, end_date)
        gtb.fill()
        self._ground_truth_bank = gtb

        # db setup
        config = PasswordPSQLConnectionConfig(database=DB_CONFIG['database'],
                                              user=DB_CONFIG['user'],
                                              password=DB_CONFIG['password'])

        psqlconn = PSQLConnector(config)

        pair_resource = PairDBResource(psqlconn)

        # populate _from_db bank
        # all valid pairs
        curr_pair_dicts = pair_resource.get_current_pairs()
        pr = Pair()
        self._from_db = [pr.from_db_dict(x) for x in curr_pair_dicts]

        # get ids of paired and unpaired warnings and gt
        pair_ids, unpaired_warns, unpaired_gt = self.run_hungarian()

        # replace warnings bank and ground truths bank with unpaired warnings and ground truth
        unp_wrn = filter(lambda x: x.id in unpaired_warns, self._warnings_bank)

        unp_gt = filter(lambda x: x.id in unpaired_gt, self._ground_truth_bank)

        # reset valid flag on all existing pairings
        pair_resource.set_valid_false()

        # report paired warnings and ground truth with scores to db
        # paired
        map(lambda x: pair_resource.sql_insert(x, psqlconn), self._bank.trim(pair_ids))

        # calculate recall
        recall = len(pair_ids)/(len(pair_ids) + len(unpaired_gt))

        # calculate precision
        # should these include warns and gts from the previous month?
        precision = len(pair_ids)/(len(pair_ids) + len(unpaired_warns))

        # build unpaired bank
        unpaired_list = unp_wrn + unp_gt

        unpaired_bank = PairBank()
        unpaired_bank._bank = unpaired_list

        return self._bank, unpaired_bank, recall, precision