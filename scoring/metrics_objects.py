"""Metric objects module."""
# pylint: disable=invalid-name,too-many-instance-attributes


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


def if_exists(dct, fld_name, ls=True):
    def joinNA(fld, delim):
        try: return None if fld is None else delim.join(fld)
        except TypeError: 
            return None
    if ls:
        try:
            if isinstance(dct[fld_name], list):
                return joinNA(dct[fld_name], ' ')
            else:
                return dct[fld_name].replace(' ', '_') if dct[fld_name] is not None else None
        except KeyError:
            return None
        else:
            try:
                return dct[fld_name]
            except KeyError:
                return None
    else:
        try:
            return dct[fld_name]
        except KeyError:
            return None


class MetricGroundTruth(object):
    """Metric ground truth class."""

    def __init__(self):
        """Set the values of this warning event."""
        self._id = None
        self._event_type = None
        self._event_subtype = None
        self._reported = None
        self._occurred = None
        self._target_industry = None
        self._target_organization = None
        self._target_entity = None
        self._threat_designation_type = None
        self._threat_designation_family = None
        self._detector_classification = None
        self._email_subject = None
        self._email_sender = None
        self._files_filename = None
        self._files_path = None
        self._files_hash = None
        self._addresses_url = None
        self._addresses_ip = None

    id = property(fget=attrgetter("_id"), fdel=attrdeleter("_id"))

    @id.setter
    def id(self, value):
        """Set id."""
        self._id = value

    event_type = property(fget=attrgetter("_event_type"), fdel=attrdeleter("_event_type"))

    @event_type.setter
    def event_type(self, value):
        """Set event type."""
        self._event_type = value

    event_subtype = property(fget=attrgetter("_event_subtype"), fdel=attrdeleter("_event_subtype"))

    @event_subtype.setter
    def event_subtype(self, value):
        """Set event subtype."""
        self._event_subtype = value

    reported = property(fget=attrgetter("_reported"), fdel=attrdeleter("_reported"))

    @reported.setter
    def reported(self, value):
        """Set reported."""
        self._reported = value

    occurred = property(fget=attrgetter("_occurred"), fdel=attrdeleter("_occurred"))

    @occurred.setter
    def occurred(self, value):
        """Set occurred."""
        self._occurred = value

    target_industry = property(fget=attrgetter("_target_industry"), fdel=attrdeleter("_target_industry"))

    @target_industry.setter
    def target_industry(self, value):
        """Set target industry."""
        self._target_industry = value

    target_organization = property(fget=attrgetter("_target_organization"), fdel=attrdeleter("_target_organization"))

    @target_organization.setter
    def target_organization(self, value):
        """Set target organization."""
        self._target_organization = value

    target_entity = property(fget=attrgetter("_target_entity"), fdel=attrdeleter("_target_entity"))

    @target_entity.setter
    def target_entity(self, value):
        """Set target entity."""
        self._target_entity = value

    threat_designation_type = property(fget=attrgetter("_threat_designation_type"),
                                       fdel=attrdeleter("_threat_designation_type"))

    @threat_designation_type.setter
    def threat_designation_type(self, value):
        """Set threat designation type."""
        self._threat_designation_type = value

    threat_designation_family = property(fget=attrgetter("_threat_designation_family"),
                                         fdel=attrdeleter("_threat_designation_family"))

    @threat_designation_family.setter
    def threat_designation_family(self, value):
        """Set threat designation family."""
        self._threat_designation_family = value

    detector_classification = property(fget=attrgetter("_detector_classification"),
                                       fdel=attrdeleter("_detector_classification"))

    @detector_classification.setter
    def detector_classification(self, value):
        """Set detector classification."""
        self._detector_classification = value

    email_subject = property(fget=attrgetter("_email_subject"), fdel=attrdeleter("_email_subject"))

    @email_subject.setter
    def email_subject(self, value):
        """Set email subject."""
        self._email_subject = value

    email_sender = property(fget=attrgetter("_email_sender"), fdel=attrdeleter("_email_sender"))

    @email_sender.setter
    def email_sender(self, value):
        """Set sender."""
        self._email_sender = value

    files_filename = property(fget=attrgetter("_files_filename"), fdel=attrdeleter("_files_filename"))

    @files_filename.setter
    def files_filename(self, value):
        """Set files filename."""
        self._files_filename = value

    files_path = property(fget=attrgetter("_files_path"), fdel=attrdeleter("_files_path"))

    @files_path.setter
    def files_path(self, value):
        """Set files path."""
        self._files_path = value

    files_hash = property(fget=attrgetter("_files_hash"), fdel=attrdeleter("_files_hash"))

    @files_hash.setter
    def files_hash(self, value):
        """Set files hash."""
        self._files_hash = value

    addresses_url = property(fget=attrgetter("_addresses_url"), fdel=attrdeleter("_addresses_url"))

    @addresses_url.setter
    def addresses_url(self, value):
        """Set addresses URL."""
        self._addresses_url = value

    addresses_ip = property(fget=attrgetter("_addresses_ip"), fdel=attrdeleter("_addresses_ip"))

    @addresses_ip.setter
    def addresses_ip(self, value):
        """Set addresses IP."""
        self._addresses_ip = value

    def to_dict(self):
        """Return dictionary of object."""
        return dict([(x.replace('_', ''), self.__dict__[x]) for x in self.__dict__])

    @classmethod
    def from_dict(cls, info_dict):
        """Instantiate object from dictionary.

        :param info_dict: a dictionary containing the required fields for creating a MetricGroundTruth object
        :type info_dict: dict
        """
        gt = cls()

        gt.id = if_exists(info_dict, 'id', ls=False)
        gt.event_type = if_exists(info_dict, 'event_type', ls=False)
        gt.event_subtype = if_exists(info_dict, 'event_subtype', ls=False)
        gt.reported = if_exists(info_dict, 'reported', ls=False)
        gt.occurred = if_exists(info_dict, 'occurred', ls=False)
        gt.target_industry = if_exists(info_dict, 'target_industry')
        gt.target_organization = if_exists(info_dict, 'target_organization')
        gt.target_entity = if_exists(info_dict, 'target_entity')
        gt.threat_designation_type = if_exists(info_dict, 'threat_designation_type')
        gt.threat_designation_family = if_exists(info_dict, 'threat_designation_family')
        gt.detector_classification = if_exists(info_dict, 'detector_classification')
        gt.email_subject = if_exists(info_dict, 'email_subject', ls=False)
        gt.email_sender = if_exists(info_dict, 'email_sender')

        # srawls: modified by adding ls=False, because the formatting_functions file already takes care of turning list into string
        gt.files_filename = if_exists(info_dict, 'files_filename', ls=False)
        gt.files_path = if_exists(info_dict, 'files_path', ls=False)
        gt.files_hash = if_exists(info_dict, 'files_hash', ls=False)
        gt.addresses_url = if_exists(info_dict, 'addresses_url', ls=False)
        gt.addresses_ip = if_exists(info_dict, 'addresses_ip', ls=False)

        return gt


class MetricWarning(object):
    """Metric warning class."""

    def __init__(self):
        """Set the values of this warning event."""
        self._id = None
        self._event_type = None
        self._event_subtype = None
        self._submitted = None
        self._probability = None
        self._occurred = None
        self._target_industry = None
        self._target_organization = None
        self._target_entity = None
        self._threat_designation_type = None
        self._threat_designation_family = None
        self._detector_classification = None
        self._email_subject = None
        self._email_sender = None
        self._files_filename = None
        self._files_path = None
        self._files_hash = None
        self._addresses_url = None
        self._addresses_ip = None

    id = property(fget=attrgetter("_id"), fdel=attrdeleter("_id"))

    @id.setter
    def id(self, value):
        """Set id."""
        self._id = value

    event_type = property(fget=attrgetter("_event_type"), fdel=attrdeleter("_event_type"))

    @event_type.setter
    def event_type(self, value):
        """Set event type."""
        self._event_type = value

    event_subtype = property(fget=attrgetter("_event_subtype"), fdel=attrdeleter("_event_subtype"))

    @event_subtype.setter
    def event_subtype(self, value):
        """Set event subtype."""
        self._event_subtype = value

    submitted = property(fget=attrgetter("_submitted"), fdel=attrdeleter("_submitted"))

    @submitted.setter
    def submitted(self, value):
        """Set submitted."""
        self._submitted = value

    probability = property(fget=attrgetter("_probability"), fdel=attrdeleter("_probability"))

    @probability.setter
    def probability(self, value):
        """Set probability."""
        self._probability = value

    occurred = property(fget=attrgetter("_occurred"), fdel=attrdeleter("_occurred"))

    @occurred.setter
    def occurred(self, value):
        """Set occurred."""
        self._occurred = value

    target_industry = property(fget=attrgetter("_target_industry"), fdel=attrdeleter("_target_industry"))

    @target_industry.setter
    def target_industry(self, value):
        """Set target industry."""
        self._target_industry = value

    target_organization = property(fget=attrgetter("_target_organization"), fdel=attrdeleter("_target_organization"))

    @target_organization.setter
    def target_organization(self, value):
        """Set target organization."""
        self._target_organization = value

    target_entity = property(fget=attrgetter("_target_entity"), fdel=attrdeleter("_target_entity"))

    @target_entity.setter
    def target_entity(self, value):
        """Set target entity."""
        self._target_entity = value

    threat_designation_type = property(fget=attrgetter("_threat_designation_type"),
                                       fdel=attrdeleter("_threat_designation_type"))

    @threat_designation_type.setter
    def threat_designation_type(self, value):
        """Set threat designation type."""
        self._threat_designation_type = value

    threat_designation_family = property(fget=attrgetter("_threat_designation_family"),
                                         fdel=attrdeleter("_threat_designation_family"))

    @threat_designation_family.setter
    def threat_designation_family(self, value):
        """Set threat designation family."""
        self._threat_designation_family = value

    detector_classification = property(fget=attrgetter("_detector_classification"),
                                       fdel=attrdeleter("_detector_classification"))

    @detector_classification.setter
    def detector_classification(self, value):
        """Set detector classification."""
        self._detector_classification = value

    email_subject = property(fget=attrgetter("_email_subject"), fdel=attrdeleter("_email_subject"))

    @email_subject.setter
    def email_subject(self, value):
        """Set email subject."""
        self._email_subject = value

    email_sender = property(fget=attrgetter("_email_sender"), fdel=attrdeleter("_email_sender"))

    @email_sender.setter
    def email_sender(self, value):
        """Set email sender."""
        self._email_sender = value

    files_filename = property(fget=attrgetter("_files_filename"), fdel=attrdeleter("_files_filename"))

    @files_filename.setter
    def files_filename(self, value):
        """Set files filename."""
        self._files_filename = value

    files_path = property(fget=attrgetter("_files_path"), fdel=attrdeleter("_files_path"))

    @files_path.setter
    def files_path(self, value):
        """Set files path."""
        self._files_path = value

    files_hash = property(fget=attrgetter("_files_hash"), fdel=attrdeleter("_files_hash"))

    @files_hash.setter
    def files_hash(self, value):
        """Set files hash."""
        self._files_hash = value

    addresses_url = property(fget=attrgetter("_addresses_url"), fdel=attrdeleter("_addresses_url"))

    @addresses_url.setter
    def addresses_url(self, value):
        """Set addresses url."""
        self._addresses_url = value

    addresses_ip = property(fget=attrgetter("_addresses_ip"), fdel=attrdeleter("_addresses_ip"))

    @addresses_ip.setter
    def addresses_ip(self, value):
        """Set addresses IP."""
        self._addresses_ip = value

    def to_dict(self):
        """Return object as a dictionary."""
        return dict([(x.replace('_', ''), self.__dict__[x]) for x in self.__dict__])

    @classmethod
    def from_dict(cls, info_dict):
        """Instantiate object from dictionary.

        :param info_dict: a dictionary containing the required fields for creating a MetricGroundTruth object
        :type info_dict: dict
        """
        warn = cls()

        warn.id = if_exists(info_dict, 'id', ls=False)
        warn.event_type = if_exists(info_dict, 'event_type', ls=False)
        warn.event_subtype = if_exists(info_dict, 'event_subtype', ls=False)
        warn.submitted = if_exists(info_dict, 'submitted', ls=False)
        warn.probability = if_exists(info_dict, 'probability', ls=False)
        warn.occurred = if_exists(info_dict, 'occurred', ls=False)
        warn.target_industry = if_exists(info_dict, 'target_industry')
        warn.target_organization = if_exists(info_dict, 'target_organization')
        warn.target_entity = if_exists(info_dict, 'target_entity')
        warn.threat_designation_type = if_exists(info_dict, 'threat_designation_type')
        warn.threat_designation_family = if_exists(info_dict, 'threat_designation_family')
        warn.detector_classification = if_exists(info_dict, 'detector_classification')
        warn.email_subject = if_exists(info_dict, 'email_subject', ls=False)
        warn.email_sender = if_exists(info_dict, 'email_sender')

        # srawls: modified by adding ls=False, because the formatting_functions file already takes care of turning list into string
        warn.files_filename = if_exists(info_dict, 'files_filename', ls=False)
        warn.files_path = if_exists(info_dict, 'files_path', ls=False)
        warn.files_hash = if_exists(info_dict, 'files_hash', ls=False)
        warn.addresses_url = if_exists(info_dict, 'addresses_url', ls=False)
        warn.addresses_ip = if_exists(info_dict, 'addresses_ip', ls=False)

        return warn
