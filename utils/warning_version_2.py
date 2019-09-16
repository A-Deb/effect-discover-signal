import numpy as np
from time import time
import pdb
import codecs
import json

target_industry = {"dexter": "financial-services",
                   "armstrong": "defense-industrial-base",
                   "knox": "defense-industrial-base"}

email_subtypeD = {0: "with-link", 1: "with-attachment",
                  2: "with-link-and-attachment"}
email_subtype_prob = {"dexter": [11.0 / 114, 31.0 / 114, 72.0 / 114],
                      "armstrong": [32.0 / 41, 3.0 / 41, 6.0 / 41],
                      "knox": [1.0 / 3, 1.0 / 3, 1.0 / 3]}

email_detector_class = {0: "Phish.URL", 1: "Malware.Binary.Doc", 2: "Unknown",
                        3: "generic"}
email_detector_class_prob = {
    "dexter": [18.0 / 31.0, 12.0 / 31.0, 1.0 / 31.0, 0.0],
    "armstrong": [0, 0, 0, 1.0],
    "knox": [1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4]}

email_threat_designation_type = {0: "trojan", 1: "other", 2: "malware",
                                 3: "pup", 4: "exploit_kit", 5: "malemail",
                                 6: "ransomware", 7: "rat", 8: "virus"}
email_threat_designation_type_prob = {
    "armstrong": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    "dexter": [1 / 9., 1 / 9., 1 / 9., 1 / 9., 1 / 9., 1 / 9., 1 / 9., 1 / 9.,
               1 / 9.],
    "knox": [1 / 9., 1 / 9., 1 / 9., 1 / 9., 1 / 9., 1 / 9., 1 / 9., 1 / 9.,
             1 / 9.]
}

email_threat_designation_family = {0: "unknown", 1: "js/redirect",
                                   2: "neutrino", 3: "other"}
email_threat_designation_family_prob = {
    "armstrong": [1, 0, 0, 0],
    "dexter": [0.7, 0.1, 0.1, 0.1],
    "knox": [1, 0, 0, 0]
}

mal_dest_threat_desig_fam = {0: "unknown", 1: "dridex", 2: None}
mal_dest_threat_desig_fam_prob = {
    "armstrong": [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0],
    "dexter": [0.8, 0.1, 0.1],
    "knox": [1.0 / 3, 1.0 / 3, 1.0 / 3]
}

mal_dest_designation_type = {0: "trojan", 1: "exploit_kit", 2: "malware",
                             3: "pup", 4: "other"}
mal_dest_designation_type_prob = {
    "armstrong": [5.0 / 9.0, 2.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 0],
    "dexter": [0.2, 0.2, 0.2, 0.2, 0.2],
    "knox": [1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5]
}

mal_dest_detector_class = {0: "Trojan.Dridex", 1: "Trojan",
                           2: "Exploit.Kit.Angler",
                           3: "Exploit.Kit.Redirect"}
mal_dest_detector_class_prob = {"armstrong": [.5, .5, 0, 0],
                                "dexter": [0, 0, 47.0 / 58.0, 11.0 / 58.0],
                                "knox": [1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4]
                                }

malware_target_entity = {0: "goreer", 1: "angelinia", 2: "smithva",
                         3: "unknown"}
malware_target_entity_prob = {
    "armstrong": np.array([3.0 / 7.0, 2.0 / 7.0, 2.0 / 7.0, 0]),
    "dexter": np.array([0, 0, 0, 1.0]),
    "knox": np.array([0, 0, 0, 1.0])
}

malware_detector_classification = {0: "US/Redirector.db",
                                   1: "HTML/neutrinp.b",
                                   2: "Exploit-SWF.bd",
                                   3: "US/Redirector.cy", 4: "Gamarue-FDL",
                                   5: "Artemis"}
malware_detector_classification_prob = {
    "armstrong": [8.0 / 26.0, 6.0 / 26.0, 6.0 / 26.0, 6.0 / 26.0, 0, 0],
    "dexter": [0, 0, 0, 0, 13.0 / 22.0, 9.0 / 22.0],
    "knox": [1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6]
}

malware_threat_designation_type = {0: "trojan", 1: "other", 2: "malware",
                                   3: "pup"}
malware_threat_designation_type_prob = {
    "armstrong": [28.0 / 70.0, 23.0 / 70.0, 12.0 / 70.0, 7.0 / 70.0],
    "dexter": [0.1, 0.1, 0.8, 0],
    "knox": [1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4]
}

malware_threat_designation_family = {0: "unknown", 1: "js/redirect",
                                     2: "neutrino", 3: "other"}
malware_threat_designation_family_prob = {
    "armstrong": [23.0 / 60.0, 19.0 / 60.0, 10.0 / 60.0, 8.0 / 60.0],
    "dexter": [0.4, 0.3, 0.0, 0.3],
    "knox": [1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4]
}


def fill_common_fields_warn_obj(warning_object, output_name, model_id,
                                created_time):
    warning_object["output_name"] = output_name
    warning_object["output_type"] = "candidate_warning"
    warning_object["audit_trail_id"] = None
    warning_object["processed_by"] = model_id
    warning_object["created"] = created_time
    return warning_object


def fill_common_fields_warnings(warning, target_organization, event_type, idx,
                                created_time, predicted_time, model_id,
                                warn_probability=0.75):
    # warning["id"] = idx
    warning["id"] = model_id + "_" + created_time + "_" + str(idx)
    warning["version"] = 2
    warning["event_type"] = event_type
    warning["occurred"] = predicted_time
    # warning["reported"] = predicted_time
    warning["probability"] = warn_probability
    
    # warning["target_industry"] = target_industry[target_organization]
    warning["target_industry"] = 'defense-industrial-base'
    warning["target_organization"] = target_organization
    return warning


def sample_evs(ev_entity, count, ev_entity_prob, target_organization):
    try:
        # print(ev_entity.keys())
        # print(count)
        # print(ev_entity_prob[target_organization])
        return np.random.choice(list(ev_entity.keys()), count,
                                p=ev_entity_prob[target_organization],
                                replace=False)
    except:
        # print("Error occurred in sample_evs")
        return np.random.choice(list(ev_entity.keys()), count,
                                p=ev_entity_prob[target_organization],
                                replace=True)


def sample_from_distribution(value_names, count, value_prob):
    return np.random.choice(value_names, count, p=value_prob, replace=True)


def sample_mal_email_events(warnings, target_organization, count):
    email_subtype_samples = sample_evs(email_subtypeD, count,
                                       email_subtype_prob, target_organization)
    email_threat_designation_type_samples = sample_evs(
        email_threat_designation_type,
        count,
        email_threat_designation_type_prob,
        target_organization)
    email_threat_desig_fam_samples = sample_evs(email_threat_designation_family,
                                                count,
                                                email_threat_designation_family_prob,
                                                target_organization)
    detector_classification_samples = sample_evs(email_detector_class, count,
                                                 email_detector_class_prob,
                                                 target_organization)
    for warning_id in range(count):
        warnings[warning_id]["event_subtype"] = email_subtypeD[
            email_subtype_samples[
                warning_id]]  # email_subtypeD[np.where(np.random.multinomial(1, email_subtype_prob[target_organization])> 0)[0][0]]
        warnings[warning_id]["target_entity"] = "user@domain"
        warnings[warning_id]["threat_designation_type"] = \
            email_threat_designation_type[
                email_threat_designation_type_samples[warning_id]]
        warnings[warning_id]["threat_designation_family"] = \
            email_threat_designation_family[
                email_threat_desig_fam_samples[warning_id]]
        warnings[warning_id]["detector_classification"] = email_detector_class[
            detector_classification_samples[
                warning_id]]  # email_detector_class[np.where(np.random.multinomial(1, email_detector_class_prob[target_organization])> 0)[0][0]]
    return warnings


# def sample_mal_email_events_knox(warnings, count,
#     freq_dist_file = "data/knox/knox-distribution_malicious-email.json"):

def sample_events_knox(warnings, count, event_type, freq_dist_file):
    # print("Loading:", freq_dist_file)
    freq_dist = json.load(codecs.open(freq_dist_file))
    # print(freq_dist.keys())
    # print(type(freq_dist))
    # 'detector_classification', 'threat_designation_family', 'event_subtype', 'threat_designation_type'

    if event_type == "malicious-email":
        event_subtype = freq_dist['event_subtype']
        keys = list(event_subtype.keys())
        values = event_subtype.values()
        prob_event_subtype = np.array([float(x) / sum(values) for x in values])

        event_subtype_samples = sample_from_distribution(keys, count,
                                                         prob_event_subtype)
    
    target_entity = freq_dist['target_entity']
    keys = list(target_entity.keys())
    values = target_entity.values()
    prob_target_entity = np.array(
        [float(x) / sum(values) for x in values])
    event_target_entity_samples = sample_from_distribution(
        keys, count, prob_target_entity)
    
    threat_designation_type = freq_dist['threat_designation_type']
    keys = list(threat_designation_type.keys())
    values = threat_designation_type.values()
    prob_threat_designation_type = np.array(
        [float(x) / sum(values) for x in values])

    event_threat_designation_type_samples = sample_from_distribution(
        keys, count, prob_threat_designation_type)

    threat_designation_family = freq_dist['threat_designation_family']
    keys = list(threat_designation_family.keys())
    values = threat_designation_family.values()
    prob_threat_designation_family = np.array(
        [float(x) / sum(values) for x in values])

    event_threat_desig_family_samples = sample_from_distribution(
        keys, count, prob_threat_designation_family)

    threat_detector_classification = freq_dist['detector_classification']
    keys = list(threat_detector_classification.keys())
    values = threat_detector_classification.values()
    prob_threat_detector_classification = np.array(
        [float(x) / sum(values) for x in values])
    event_detector_classification_samples = sample_from_distribution(
        keys, count, prob_threat_detector_classification)

    for warning_id in range(count):
        if event_type == "malicious-email":
            warnings[warning_id]["event_subtype"] = event_subtype_samples[
                warning_id]  # email_subtypeD[np.where(np.random.multinomial(1, email_subtype_prob[target_organization])> 0)[0][0]]
            warnings[warning_id]["target_entity"] = event_target_entity_samples[warning_id]
        warnings[warning_id]["threat_designation_type"] = \
            event_threat_designation_type_samples[warning_id]
        warnings[warning_id]["threat_designation_family"] = \
            event_threat_desig_family_samples[warning_id]
        warnings[warning_id]["detector_classification"] = \
            event_detector_classification_samples[
                warning_id]  # email_detector_class[np.where(np.random.multinomial(1, email_detector_class_prob[target_organization])> 0)[0][0]]

    return warnings


def sample_mal_dest_and_malware_events_knox(
        warnings, count, freq_dist_file=
        "data/knox/knox-distribution_malicious-email.json"):
    freq_dist = json.load(codecs.open(freq_dist_file))

    event_subtype = freq_dist['event_subtype']
    keys = list(event_subtype.keys())
    values = event_subtype.values()
    prob_event_subtype = np.array([float(x) / sum(values) for x in values])

    event_subtype_samples = sample_from_distribution(keys, count,
                                                     prob_event_subtype)

    threat_designation_type = freq_dist['threat_designation_type']
    keys = list(threat_designation_type.keys())
    values = threat_designation_type.values()
    prob_threat_designation_type = np.array(
        [float(x) / sum(values) for x in values])

    event_threat_designation_type_samples = sample_from_distribution(
        keys, count, prob_threat_designation_type)

    threat_designation_family = freq_dist['threat_designation_family']
    keys = list(threat_designation_family.keys())
    values = threat_designation_family.values()
    prob_threat_designation_family = np.array(
        [float(x) / sum(values) for x in values])

    event_threat_desig_family_samples = sample_from_distribution(
        keys, count, prob_threat_designation_family)

    threat_detector_classification = freq_dist['detector_classification']
    keys = list(threat_detector_classification.keys())
    values = threat_detector_classification.values()
    prob_threat_detector_classification = np.array(
        [float(x) / sum(values) for x in values])
    event_detector_classification_samples = sample_from_distribution(
        keys, count, prob_threat_detector_classification)

    for warning_id in range(count):
        warnings[warning_id]["event_subtype"] = event_subtype_samples[
            warning_id]  # email_subtypeD[np.where(np.random.multinomial(1, email_subtype_prob[target_organization])> 0)[0][0]]
        warnings[warning_id]["target_entity"] = "user@domain"
        warnings[warning_id]["threat_designation_type"] = \
            event_threat_designation_type_samples[warning_id]
        warnings[warning_id]["threat_designation_family"] = \
            event_threat_desig_family_samples[warning_id]
        warnings[warning_id]["detector_classification"] = \
            event_detector_classification_samples[
                warning_id]  # email_detector_class[np.where(np.random.multinomial(1, email_detector_class_prob[target_organization])> 0)[0][0]]

    return warnings


def sample_mal_dest_events(warnings, target_organization, count):
    mal_dest_designation_type_samples = sample_evs(mal_dest_designation_type,
                                                   count,
                                                   mal_dest_designation_type_prob,
                                                   target_organization)
    mal_dest_threat_desig_fam_samples = sample_evs(mal_dest_threat_desig_fam,
                                                   count,
                                                   mal_dest_threat_desig_fam_prob,
                                                   target_organization)
    mal_dest_detector_class_samples = sample_evs(mal_dest_detector_class, count,
                                                 mal_dest_detector_class_prob,
                                                 target_organization)
    for warning_id in range(count):
        warnings[warning_id]["threat_designation_type"] = \
            mal_dest_designation_type[mal_dest_designation_type_samples[
                warning_id]]  # mal_dest_designation_type[np.where(np.random.multinomial(1, mal_dest_designation_type_prob[target_organization])> 0)[0][0]]
        warnings[warning_id]["threat_designation_family"] = \
            mal_dest_threat_desig_fam[mal_dest_threat_desig_fam_samples[
                warning_id]]  # mal_dest_threat_desig_fam[np.where(np.random.multinomial(1, mal_dest_threat_desig_fam_prob[target_organization])> 0)[0][0]]
        warnings[warning_id]["detector_classification"] = \
            mal_dest_detector_class[mal_dest_detector_class_samples[
                warning_id]]  # mal_dest_detector_class[np.where(np.random.multinomial(1, mal_dest_detector_class_prob[target_organization])> 0)[0][0]]
    return warnings


def sample_malware_events(warnings, target_organization, count):
    malware_target_entity_samples = sample_evs(malware_target_entity, count,
                                               malware_target_entity_prob,
                                               target_organization)
    malware_threat_designation_type_samples = sample_evs(
        malware_threat_designation_type, count,
        malware_threat_designation_type_prob, target_organization)
    malware_threat_designation_family_samples = sample_evs(
        malware_threat_designation_family, count,
        malware_threat_designation_family_prob, target_organization)
    malware_detector_classification_samples = sample_evs(
        malware_detector_classification, count,
        malware_detector_classification_prob, target_organization)
    for warning_id in range(count):
        warnings[warning_id]["target_entity"] = malware_target_entity[
            malware_target_entity_samples[
                warning_id]]  # malware_target_entity[np.where(np.random.multinomial(1, malware_target_entity_prob[target_organization])> 0)[0][0]]
        warnings[warning_id]["threat_designation_type"] = \
            malware_threat_designation_type[
                malware_threat_designation_type_samples[
                    warning_id]]  # malware_threat_designation_type[np.where(np.random.multinomial(1, malware_threat_designation_type_prob[target_organization])> 0)[0][0]]
        warnings[warning_id]["threat_designation_family"] = \
            malware_threat_designation_family[
                malware_threat_designation_family_samples[
                    warning_id]]  # malware_threat_designation_family[np.where(np.random.multinomial(1, malware_threat_designation_family_prob[target_organization])> 0)[0][0]]
        warnings[warning_id]["detector_classification"] = \
            malware_detector_classification[
                malware_detector_classification_samples[
                    warning_id]]  # malware_detector_classification[np.where(np.random.multinomial(1, malware_detector_classification_prob[target_organization])> 0)[0][0]]
    return warnings


def generate_warning_objects(warn_object, target_organization, event_type,
                             count, prev_id, output_name, model_id,
                             warn_timestamp, predicted_time):
    import time
    # set seed for replication
    # warn_date = warn_timestamp.date()
    warn_date = warn_timestamp

    seed = int(time.mktime(warn_date.timetuple()))
    np.random.seed(seed)

    created_time = warn_timestamp.isoformat() + "Z"
    count = int(count)
    warn_object = [None] * count
    for idx in range(count):
        warn_object[idx] = dict()
        warn_object[idx] = fill_common_fields_warn_obj(warn_object[idx],
                                                       output_name, model_id,
                                                       created_time)
    warnings = [None] * count
    for idx in range(count):
        warnings[idx] = dict()
        warnings[idx] = fill_common_fields_warnings(warnings[idx],
                                                    target_organization,
                                                    event_type,
                                                    prev_id + idx + 1,
                                                    created_time,
                                                    predicted_time,
                                                    model_id)
    if event_type == "malicious-email":
        warnings = sample_mal_email_events(warnings, target_organization, count)
    elif event_type == "malicious-destination":
        warnings = sample_mal_dest_events(warnings, target_organization, count)
    elif event_type == "endpoint-malware":
        warnings = sample_malware_events(warnings, target_organization, count)
    for idx in range(count):
        warn_object[idx]["warning"] = warnings[idx]
    return warn_object


def generate_warning_objects_knox(warn_object, target_organization, event_type,
                                  count, prev_id, output_name, model_id,
                                  warn_timestamp, predicted_time,
                                  feature_dist_file):
    import time
    warn_date = warn_timestamp
    seed = int(time.mktime(warn_date.timetuple()))
    np.random.seed(seed)

    created_time = warn_timestamp.isoformat() + "Z"
    count = int(count)
    warn_object = [None] * count
    for idx in range(count):
        warn_object[idx] = dict()
        warn_object[idx] = fill_common_fields_warn_obj(warn_object[idx],
                                                       output_name, model_id,
                                                       created_time)
    warnings = [None] * count
    for idx in range(count):
        warnings[idx] = dict()
        warnings[idx] = fill_common_fields_warnings(warnings[idx],
                                                    target_organization,
                                                    event_type,
                                                    prev_id + idx + 1,
                                                    created_time,
                                                    predicted_time,
                                                    model_id)
    warnings = sample_events_knox(warnings, count, event_type,
                                  feature_dist_file)
    # if event_type == "malicious-email":
    #     warnings = sample_mal_email_events_knox(warnings, count, event_type)
    # elif event_type == "malicious-destination":
    #     warnings = sample_mal_dest_and_malware_events_knox(warnings, count)
    # elif event_type == "endpoint-malware":
    #     warnings = sample_mal_dest_and_malware_events_knox(warnings, count)

    # print(warnings)
    for idx in range(count):
        warn_object[idx]["warning"] = warnings[idx]
    return warn_object


if __name__ == "__main__":
    modelOut = []
    idx = 0
    for target_organization in ["dexter", "armstrong"]:
        warn_object = dict()
        warn_object = generate_warning_objects(warn_object, target_organization,
                                               "malicious-email", 3, idx,
                                               "hmm_warning",
                                               "HMM_2_Poisson_dexter_endpoint-malware",
                                               time(), time())
        idx += 3
        modelOut += warn_object
        warn_object = dict()
        warn_object = generate_warning_objects(warn_object, target_organization,
                                               "malicious-destination", 3, idx,
                                               "hmm_warning",
                                               "HMM_2_Poisson_dexter_endpoint-malware",
                                               time(), time())
        idx += 3
        modelOut += warn_object
        warn_object = dict()
        warn_object = generate_warning_objects(warn_object, target_organization,
                                               "endpoint-malware", 3, idx,
                                               "hmm_warning",
                                               "HMM_2_Poisson_dexter_endpoint-malware",
                                               time(), time())
        idx += 3
        modelOut += warn_object
    pdb.set_trace()
