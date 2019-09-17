def if_exists(j_dict, fld, second=None, join=False):
    """ To return field in a given CAUSE event dict if it exists.

    :param j_dict: CAUSE event dict
    :type j_dict: dict
    :param fld: the desired field
    :type fld: str
    :param second: if the desired field is in a address dict or file dict
                   this the desired field and fld is populated with either
                   addresses or files
    :type second: str
    :param join: if the desired field is a list this will replace the spaces
                 underscores in each item and join the list by an empty space
    :type join: bool
    """
    try:
        res = j_dict[fld]
        if second:
            if isinstance(res, list):
                res = [r[second] for r in res]
                res_list = [y for x in res for y in x if isinstance(x, list)]
                res_item = [x for x in res if not isinstance(x, list)]
                res = res_list + res_item
            else:
                res = res[second]
        if join:
            if isinstance(res, list):
                res = [r.replace(' ', '_') for r in res]
                res = ' '.join(res)
            else:
                pass
        return res
    except KeyError:
        return None


def format_gt(j_dict):
    """Returns a flattened version of a ground truth record

    :param j_dict: ground truth event dict
    :type j_dict: dict
    """
    fields = {'id': if_exists(j_dict, 'id'),
              'event_type': if_exists(j_dict, 'event_type'),
              'event_subtype': if_exists(j_dict, 'event_subtype'),
              'reported': if_exists(j_dict, 'reported'),
              'occurred': if_exists(j_dict, 'occurred'),
              'target_industry': if_exists(j_dict, 'target_industry'),
              'target_organization': if_exists(j_dict, 'target_organization'),
              # target entity should be a list but from what we have seen in the GT
              # that has been released it hasn't been
              'target_entity': if_exists(j_dict, 'target_entity'),
              'threat_designation_type': if_exists(j_dict, 'threat_designation_type', join=True),
              'threat_designation_family': if_exists(j_dict, 'threat_designation_family', join=True),
              'detector_classification': if_exists(j_dict, 'detector_classification', join=True),
              'email_subject': if_exists(j_dict, 'email_subject'),
              'email_sender': if_exists(j_dict, 'email_sender', join=True),
              'files_filename': if_exists(j_dict, 'files', second='filename', join=True),
              'files_path': if_exists(j_dict, 'files', second='path', join=True),
              'files_hash': if_exists(j_dict, 'files', second='hash', join=True),
              'addresses_url': if_exists(j_dict, 'addresses', second='url', join=True),
              'addresses_ip': if_exists(j_dict, 'addresses', second='ip', join=True),
              'version': if_exists(j_dict, 'version')}

    return fields


def format_warn(j_dict):
    """Returns a flattened version of a warning record

    :param j_dict: ground truth event dict
    :type j_dict: dict

    Note: only information that is needed for scoring is brought in
    """
    fields = {'id': if_exists(j_dict, 'id'),
              'event_type': if_exists(j_dict, 'event_type'),
              'event_subtype': if_exists(j_dict, 'event_subtype'),
              'occurred': if_exists(j_dict, 'occurred'),
              'probability': if_exists(j_dict, 'probability'),
              'target_industry': if_exists(j_dict, 'target_industry'),
              'target_organization': if_exists(j_dict, 'target_organization'),
              'target_entity': if_exists(j_dict, 'target_entity'),
              'threat_designation_type': if_exists(j_dict, 'threat_designation_type', join=True),
              'threat_designation_family': if_exists(j_dict, 'threat_designation_family', join=True),
              'detector_classification': if_exists(j_dict, 'detector_classification', join=True),
              'email_subject': if_exists(j_dict, 'email_subject'),
              'email_sender': if_exists(j_dict, 'email_sender'),
              'files_filename': if_exists(j_dict, 'files', second='filename', join=True),
              'files_path': if_exists(j_dict, 'files', second='path', join=True),
              'files_hash': if_exists(j_dict, 'files', second='hash', join=True),
              'addresses_url': if_exists(j_dict, 'addresses', second='url', join=True),
              'addresses_ip': if_exists(j_dict, 'addresses', second='ip', join=True),
              'version': if_exists(j_dict, 'version')}

    return fields
