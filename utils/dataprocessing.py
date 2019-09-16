#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import pandas as pd
from elasticsearch import Elasticsearch

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def get_connection_to_es(con_str):
    # con_str = "http://ec2-52-42-169-124.us-west-2.compute.amazonaws.com/es/"
    # con_str = "http://cloudweb01.isi.edu/es/"
    return Elasticsearch(
        [con_str],
        http_auth=('effect', 'c@use!23'), port=80)


def gt_hourly_time_series(data_source):
    if data_source == 'armstrong_endpoint-malware':
        print("Loading:", data_source)
        ts = load_armstrong_hourly(event_type='endpoint-malware')
        es_query = {}
    elif data_source == 'armstrong_malicious-destination':
        print("Loading:", data_source)
        ts = load_armstrong_hourly(
            event_type='malicious-destination')
        es_query = {}
    elif data_source == 'armstrong_malicious-email':
        print("Loading:", data_source)
        ts = load_armstrong_hourly(event_type='malicious-email')
        es_query = {}
    elif data_source == 'knox_endpoint-malware':
        print("Loading:", data_source)
        ts = load_knox_hourly(event_type='endpoint-malware')
        es_query = {}
    elif data_source == 'knox_malicious-destination':
        print("Loading:", data_source)
        ts = load_knox_hourly(event_type='malicious-destination')
        es_query = {}
    elif data_source == 'knox_malicious-email':
        print("Loading:", data_source)
        ts = load_knox_hourly(event_type='malicious-email')
        es_query = {}
    else:
        raise Exception("Not a valid data source: {}".format(data_source))
    return ts, es_query


def gt_time_series(options):
    ts = None
    es_query = None

    gt_index = "effect-gt"

    if options.data_replication:
        gt_index = gt_index + "-" + (
            options.warn_start_date + pd.Timedelta(days=-1)).strftime("%Y%m%d")
        print("Replication index: ", gt_index)

    if options.data_source == 'ransomware_locky':
        print("Fetching data from elastic search")
        print("\tData source:", options.data_source)
        ts, es_query = get_ransom_data_elastic_search(
            windowsize=options.window_size, malwaretype="Locky",
            end_date=options.warn_start_date + pd.Timedelta(days=-1),
            hostname=options.data_hostname)
        # ylabel = "Num of Attacks"
        # offset = 200
        # state_level = 50

    elif options.data_source == 'ransomware_cerber':
        print("Fetching data from elastic search")
        print("\tData source:", options.data_source)
        ts, es_query = get_ransom_data_elastic_search(
            windowsize=options.window_size, malwaretype="Cerber",
            end_date=options.warn_start_date + pd.Timedelta(days=-1),
            hostname=options.data_hostname)
        # ylabel = "Num of Attacks"
        # offset = 40
        # state_level = 5
    elif options.data_source == 'dexter_endpoint-malware':
        print("Loading:", options.data_source)
        # ts = load_dexter_endpoint_malware()
        # es_query = {}
        ts, es_query = get_ground_truth_ts_es(company="dexter",
                                              event_type="endpoint-malware",
                                              hostname=options.data_hostname,
                                              index=gt_index)
    elif options.data_source == 'dexter_malicious-email':
        print("Loading:", options.data_source)
        # ts = load_dexter_malicious_email()
        # es_query = {}
        ts, es_query = get_ground_truth_ts_es(company="dexter",
                                              event_type="malicious-email",
                                              hostname=options.data_hostname,
                                              index=gt_index)
    elif options.data_source == 'dexter_malicious-destination':
        print("Loading:", options.data_source)
        # ts = load_dexter_malicious_destination()
        # es_query = {}
        ts, es_query = get_ground_truth_ts_es(company="dexter",
                                              event_type="malicious-destination",
                                              hostname=options.data_hostname,
                                              index=gt_index)

    elif options.data_source == 'armstrong_internal-data':
        print("Loading:", options.data_source)
        ts = load_armstrong_internal()
        es_query = {}


    elif options.data_source == 'armstrong_endpoint-malware':
        print("Loading:", options.data_source)
        ts = load_armstrong_endpoint_malware()
        es_query = {}
        # ts, es_query = get_ground_truth_ts_es(company="armstrong",
        #                                       event_type="endpoint-malware",
        #                                       hostname=options.data_hostname,
        #                                       index=gt_index)
    elif options.data_source == 'armstrong_malicious-destination':
        print("Loading:", options.data_source)
        ts = load_armstrong_malicious_destination()
        es_query = {}
        # ts, es_query = get_ground_truth_ts_es(company="armstrong",
        #                                       event_type="malicious-destination",
        #                                       hostname=options.data_hostname,
        #                                       index=gt_index)
    elif options.data_source == 'armstrong_malicious-email':
        print("Loading:", options.data_source)
        ts = load_armstrong_malicious_email()
        es_query = {}
        # ts, es_query = get_ground_truth_ts_es(company="armstrong",
        #                                       event_type="malicious-email",
        #                                       hostname=options.data_hostname,
        #                                       index=gt_index)
    elif options.data_source == 'knox_endpoint-malware':
        print("Loading:", options.data_source)
        ts = load_knox_endpoint_malware()
        es_query = {}
        # ts, es_query = get_ground_truth_ts_es(company="knox",
        #                                       event_type="endpoint-malware",
        #                                       hostname=options.data_hostname,
        #                                       index=gt_index)
    elif options.data_source == 'knox_malicious-destination':
        print("Loading:", options.data_source)
        ts = load_knox_malicious_destination()
        es_query = {}
        # ts, es_query = get_ground_truth_ts_es(company="knox",
        #                                       event_type="malicious-destination",
        #                                       hostname=options.data_hostname,
        #                                       index=gt_index)
    elif options.data_source == 'knox_malicious-email':
        print("Loading:", options.data_source)
        ts = load_knox_malicious_email()
        es_query = {}
        # ts, es_query = get_ground_truth_ts_es(company="knox",
        #                                       event_type="malicious-email",
        #                                       hostname=options.data_hostname,
        #                                       index=gt_index)
    else:
        print(options)
        # sys.exit("Error: Valid options are not given. Program terminated.")
    return ts, es_query


def get_ground_truth_ts_es(company, event_type,
                           hostname='http://cloudweb01.isi.edu/es/',
                           index="effect-gt-20180228"):
    con = get_connection_to_es(hostname)
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match_phrase": {"company": company},
                    },
                    {
                        "match_phrase": {"event_type": event_type},
                    }
                ]
            }
        },
        "size": 500
    }

    results = con.search(index=index, body=body)
    hits = results['hits']
    entries = hits['hits']
    # print(hits.keys())
    num_entries = hits['total']
    print("Total num of entries =", num_entries, len(entries))

    date = []
    count = []
    for entry in entries:
        # date.append(pd.to_datetime(entry['_source']['date']).date())
        date.append(pd.to_datetime(entry['_source']['date']))
        count.append(entry['_source']['count'])

    ts = pd.Series(count, index=date)
    ts.name = 'count'
    ts.index.name = 'date'
    min_date = ts.index.min()
    max_date = ts.index.max()
    idx = pd.date_range(min_date, max_date)
    ts = ts.reindex(index=idx, fill_value=0)

    return ts, body


def gt_time_series_from_file(options):
    ts = None
    if options.data_source == 'dexter_endpoint-malware':
        print("Loading:", options.data_source)
        ts = load_dexter_endpoint_malware()
    elif options.data_source == 'dexter_malicious-email':
        print("Loading:", options.data_source)
        ts = load_dexter_malicious_email()
    elif options.data_source == 'dexter_malicious-destination':
        print("Loading:", options.data_source)
        ts = load_dexter_malicious_destination()
    elif options.data_source == 'armstrong_endpoint-malware':
        print("Loading:", options.data_source)
        ts = load_armstrong_endpoint_malware()
    elif options.data_source == 'armstrong_malicious-destination':
        print("Loading:", options.data_source)
        ts = load_armstrong_malicious_destination()
    elif options.data_source == 'armstrong_malicious-email':
        print("Loading:", options.data_source)
        ts = load_armstrong_malicious_email()
    else:
        print(options)
    return ts


def load_time_series(data_source, window_size):
    ts = None
    es_query = None
    if data_source == 'ransomware_locky':
        print("Fetching data from elastic search")
        print("\tData source:", data_source)
        ts, es_query = get_ransom_data_elastic_search(
            windowsize=window_size, malwaretype="Locky")
    elif data_source == 'ransomware_cerber':
        print("Fetching data from elastic search")
        print("\tData source:", data_source)
        ts, es_query = get_ransom_data_elastic_search(
            windowsize=window_size, malwaretype="Cerber")
    else:
        print("Error: Valid options are not given")
    return ts, es_query


# <editor-fold desc="ES query body">
def get_body_given_duration(start_date, end_date, start_indx=0,
                            payload_size=50):
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "observedDate": {
                                "gte": start_date.strftime("%d/%m/%Y"),
                                "lte": end_date.strftime("%d/%m/%Y"),
                                "format": "dd/MM/yyyy"
                            }
                        }
                    }
                ]
            }
        },
        "from": start_indx,
        "size": payload_size
    }
    return body


def get_body(start_date, end_date, malwaretype, start_indx=0, payload_size=10):
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "name": malwaretype
                        }
                    },
                    {
                        "range": {
                            "observedDate": {
                                "gte": start_date.strftime("%d/%m/%Y"),
                                "lte": end_date.strftime("%d/%m/%Y"),
                                "format": "dd/MM/yyyy"
                            }
                        }
                    }
                ]
            }
        },
        "from": start_indx,
        "size": payload_size
    }
    return body


def get_body_given_publisher(start_date, end_date, publisher, start_indx=0,
                             payload_size=50):
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match_phrase": {
                            "publisher": publisher
                        }
                    },
                    {
                        "range": {
                            "observedDate": {
                                "gte": start_date.strftime("%d/%m/%Y"),
                                "lte": end_date.strftime("%d/%m/%Y"),
                                "format": "dd/MM/yyyy"
                            }
                        }
                    }
                ]
            }
        },
        "from": start_indx,
        "size": payload_size
    }
    return body


# </editor-fold>

# <editor-fold desc="Direct extraction of TS from ES">
def get_ts_hg_taxii(publisher="hg-taxii",
                    hostname='http://cloudweb01.isi.edu/es/',
                    index='effect/malware'):
    es = get_connection_to_es(hostname)
    body = {
        "query": {
            "match_phrase": {
                "publisher": publisher
            }
        },
        "size": 0,
        "aggs": {
            "daily_event_count": {
                "date_histogram": {
                    "field": "observedDate",
                    "interval": "day"
                }
            }

        }
    }


def get_ts_malware(publisher="hg-abusech",
                   hostname='http://cloudweb01.isi.edu/es/',
                   index='effect/malware'):
    es = get_connection_to_es(hostname)
    body = {
        "query": {
            "match_phrase": {
                "publisher": publisher
            }
        },
        "size": 0,
        "aggs": {
            "daily_event_count": {
                "date_histogram": {
                    "field": "observedDate",
                    "interval": "day"
                }
            }

        }
    }

    results = es.search(index=index, body=body)
    # print(results["timed_out"])
    # print(results["_shards"]["successful"])
    # print(results["hits"]["total"])
    # print(len(results["aggregations"]["daily_event_count"]["buckets"]))

    df = pd.DataFrame(results["aggregations"]["daily_event_count"]["buckets"])
    # print(df.head())
    df.rename(columns={"key_as_string": "date", "doc_count": "count"},
              inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # print(df.head())

    return df['count']


def get_ts_malware_exclue_Locky(publisher="hg-abusech",
                                hostname='http://cloudweb01.isi.edu/es/',
                                index='effect/malware'):
    es = get_connection_to_es(hostname)
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match_phrase": {
                            "publisher": publisher
                        }
                    }
                ],
                "must_not": [
                    {
                        "match_phrase": {
                            "name": "Locky"
                        }
                    }
                ]

            }
        },
        "size": 0,
        "aggs": {
            "daily_event_count": {
                "date_histogram": {
                    "field": "observedDate",
                    "interval": "day"
                }
            }
        }
    }

    results = es.search(index=index, body=body)
    # print(results["timed_out"])
    # print(results["_shards"]["successful"])
    # print(results["hits"]["total"])
    # print(len(results["aggregations"]["daily_event_count"]["buckets"]))

    df = pd.DataFrame(results["aggregations"]["daily_event_count"]["buckets"])
    # print(df.head())
    df.rename(columns={"key_as_string": "date", "doc_count": "count"},
              inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # print(df.head())

    return df['count']


def get_ts_malware_given_type(atype, publisher="hg-abusech",
                              hostname='http://cloudweb01.isi.edu/es/',
                              index='effect/malware'):
    es = get_connection_to_es(hostname)
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match_phrase": {
                            "publisher": publisher
                        }
                    },
                    {
                        "match_phrase": {
                            "name": atype
                        }
                    }
                ]
            }
        },
        "size": 0,
        "aggs": {
            "daily_event_count": {
                "date_histogram": {
                    "field": "observedDate",
                    "interval": "day"
                }
            }
        }
    }

    results = es.search(index=index, body=body)
    # print(results["timed_out"])
    # print(results["_shards"]["successful"])
    # print(results["hits"]["total"])
    # print(len(results["aggregations"]["daily_event_count"]["buckets"]))

    df = pd.DataFrame(results["aggregations"]["daily_event_count"]["buckets"])
    # print(df.head())
    df.rename(columns={"key_as_string": "date", "doc_count": "count"},
              inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # print(df.head())

    return df['count']


def get_ts_all_malware(hostname='http://cloudweb01.isi.edu/es/',
                       index='effect/malware'):
    es = get_connection_to_es(hostname)
    body = {
        "aggs": {
            "events_over_time": {
                "date_histogram": {
                    "field": "observedDate",
                    "interval": "day"
                }
            }
        }
    }

    results = es.search(index=index, body=body)
    print(results["timed_out"])
    print(results["_shards"]["successful"])
    print(results["hits"]["total"])
    print(len(results["aggregations"]["events_over_time"]["buckets"]))

    df = pd.DataFrame(results["aggregations"]["events_over_time"]["buckets"])
    # print(df.head())
    df.rename(columns={"key_as_string": "date", "doc_count": "count"},
              inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # print(df.head())
    return df['count']


# </editor-fold>

def get_all_ransom_data_elastic_search(start_date, end_date=None,
                                       hostname='http://cloudweb01.isi.edu/es/',
                                       index='effect/malware',
                                       payload_size=10,
                                       max_timeout_count=3):
    es = get_connection_to_es(hostname)
    start_date = pd.to_datetime(start_date).date()
    if end_date is None:
        end_date = pd.datetime.now().date()
    results = es.search(index=index, scroll="5m",
                        body=get_body_given_duration(
                            start_date, end_date, start_indx=0,
                            payload_size=payload_size))
    sid = results['_scroll_id']
    hits = results['hits']
    entries = hits['hits']
    num_entries = hits['total']
    print("Total num of entries =", num_entries)
    print("sid =", sid)
    print("length of batch =", len(entries))

    print(entries[0])

    i = 0
    payloads = []
    timeout_count = 0

    while num_entries > 0:
        print("Scrolling...")
        res = es.scroll(scroll_id=sid, scroll='5m')
        # Update the scroll ID
        sid = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        entries = hits['hits']
        print(entries[0])

        if res['timed_out'] == False:
            payloads.append(res)
            i = i + payload_size
            timeout_count = 0
        else:
            timeout_count = timeout_count + 1
            if timeout_count == max_timeout_count:
                sys.exit("Max time out occurred while fetching data from \
                             from Elastic Search. Program terminated")

                # malware_df = process_esearch_data(payloads)
                #
                # ts_malware = malware_df.groupby('observed_date').size()
                # learn_end_date = ts_malware.index.max()
                # learn_start_date = learn_end_date - pd.Timedelta(days=windowsize - 1)
                # mask = (ts_malware.index >= learn_start_date) & (
                #     ts_malware.index <= learn_end_date)
                # ts_malware_window = ts_malware[mask]
                # date_range = pd.date_range(learn_start_date, learn_end_date)
                # ts_malware_window = ts_malware_window.reindex(date_range, fill_value=0)


def get_ransom_data_elastic_search(windowsize=50, malwaretype="Cerber",
                                   end_date=pd.datetime.now().date(),
                                   hostname='http://cloudweb01.isi.edu/es/'):
    min_num_data_points = 100
    payload_size = 100
    max_timeout_count = 3
    data_search_window = 100
    if data_search_window < windowsize:
        data_search_window = data_search_window + windowsize

    es = get_connection_to_es(hostname)

    start_date = end_date - pd.Timedelta(days=data_search_window - 1)

    query = get_body(start_date, end_date, malwaretype)
    source_type = 'malware'
    # idx = "effect-" + end_date.strftime("%Y%m%d") + "/" + source_type
    idx = "effect/" + source_type
    results = es.search(index=idx, body=query)

    hits = results['hits']
    # print(hits.keys())
    num_entries = hits['total']
    # print("Total num of entries =", num_entries)

    # print("Number of hits =", len(hits['hits']))
    # for hit in hits['hits']:
    #     for key in hit.keys():
    #         print(key, ' -> ', hit[key])
    #         print()
    # return results['hits']

    malware_df = None
    if num_entries < min_num_data_points:
        print("Error: there are less than " + str(min_num_data_points) +
              " data points over a window of " + str(windowsize) + " days.")
        # generate a warning of low intensity
    else:
        i = 0
        payloads = []
        timeout_count = 0
        while i < num_entries:
            res = es.search(body=get_body(
                start_date, end_date, malwaretype, start_indx=i,
                payload_size=payload_size), index=idx)
            if res['timed_out'] == False:
                payloads.append(res)
                i = i + payload_size
                timeout_count = 0
            else:
                timeout_count = timeout_count + 1
                if timeout_count == max_timeout_count:
                    sys.exit("Max time out occurred while fetching data from \
                             from Elastic Search. Program terminated")

        # print("Number of payloads =", len(payloads))
        malware_df = process_esearch_data(payloads)
        ts_malware = malware_df.groupby('observed_date').size()

        learn_end_date = ts_malware.index.max()
        learn_start_date = learn_end_date - pd.Timedelta(days=windowsize - 1)
        mask = (ts_malware.index >= learn_start_date) & (
            ts_malware.index <= learn_end_date)
        ts_malware_window = ts_malware[mask]
        date_range = pd.date_range(learn_start_date, learn_end_date)
        ts_malware_window = ts_malware_window.reindex(date_range, fill_value=0)
    return ts_malware_window, query


def process_esearch_data(payloads):
    data = []
    for payload in payloads:
        # print(payload.keys())
        # dict_keys(['took', 'timed_out', '_shards', 'hits'])
        hits = payload["hits"]["hits"]  # list object
        for ahit in hits:
            # print(ahit.keys())
            # dict_keys(['_index', '_type', '_id', '_score', '_source'])
            src = ahit["_source"]
            # print(src.keys())
            # ['a', 'url', 'observedDate', 'name', 'uri']
            # ['a', 'name', 'countryOfOrigin', 'hostedAt', 'uri', 'url',
            # 'observedDate']
            if isinstance(src['name'], list):
                name = src['name'][0]
            else:
                name = src['name']

            data.append(pd.to_datetime(src["observedDate"]).date(),
                        src['publisher'], name,
                        pd.to_datetime(src["dateRecorded"]))
            # data.append((src["observedDate"], src["countryOfOrigin"],
            #              src["url"], src["hostedAt"]["name"]))

    # colnames = ["observedDate", "countryOfOrigin", "url", "name"]
    colnames = ["observed_date", "publisher", "name", "record_date"]
    return pd.DataFrame(data, columns=colnames)


def get_ransom_data(datafile='ransomware/ransomware.csv',
                    malware_type='Locky', start_day=None):
    df = pd.read_csv(datafile, header=0)
    df['Firstseen'] = pd.to_datetime(df['Firstseen'],
                                     format="%Y-%m-%d", utc=True)
    df['Date'] = df['Firstseen'].dt.date
    df.index = df['Date']

    # last three date entries: 2012-02-27, 2015-03-02, 2015-06-18
    df = df[:-3]

    # get only the 'locky' malwares
    df_malware = df[df['Malware'] == malware_type]
    ts_malware = df_malware.groupby('Date').size()
    date_range = pd.date_range(df.index.min(), df.index.max())
    ts_malware = ts_malware.reindex(date_range, fill_value=0)

    # subsetting
    if start_day is None:
        start_day = df.index.min()
    else:
        start_day = pd.to_datetime(start_day).date()
    ts_malware = ts_malware[start_day:]
    return ts_malware


def get_ransom_locky(datafile='ransomware/ransomware_locky.csv'):
    return get_ransom_data(malware_type='Locky', start_day='2016-02-16')


def get_ransom_cerber(datafile='ransomware/ransomware_cerber.csv'):
    return get_ransom_data(malware_type='Cerber', start_day='2016-06-21')


def load_data(filepath):
    data = pd.read_csv(filepath, header=0)
    # data['Firstseen'] = pd.to_datetime(data['Firstseen'],
    #                                    format="%Y-%m-%d %H:%M%S")
    data['Firstseen'] = pd.to_datetime(data['Firstseen'],
                                       format="%Y-%m-%d", utc=True)
    return data




# load armstrong internal data
def load_armstrong_internal(filepath=None):
    if filepath is None:
        filepath = "data/armstrong/gt_internal.csv"
    return pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                       squeeze=True)





# dexter data
def load_dexter_all_malware(filepath=None):
    if filepath is None:
        filepath = "data/dexter/gt_events_all.csv"
    return pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                       squeeze=True)


def load_dexter_endpoint_malware(filepath=None):
    if filepath is None:
        filepath = "data/dexter/gt_endpoint-malware.csv"
    return pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                       squeeze=True)


def load_dexter_malicious_email(filepath=None):
    if filepath is None:
        filepath = "data/dexter/gt_malicious-email.csv"
    return pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                       squeeze=True)


# def load_dexter_malicious_url(filepath=None):
#     if filepath is None:
#         filepath = "data/dexter/gt_malicious-url.csv"
#     return pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
#                        squeeze=True)

def load_dexter_malicious_destination(filepath=None):
    if filepath is None:
        filepath = "./data/dexter/gt_malicious-destination.csv"
    return pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                       squeeze=True)


def load_armstrong_endpoint_malware(filepath=None):
    if filepath is None:
        filepath = 'data/armstrong/gt_endpoint-malware.csv'
    return pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                       squeeze=True)


def load_armstrong_hourly(filepath=None, event_type="endpoint_malware",
                          freq="H"):
    if filepath is None:
        # "data/armstrong/nov-13/armstrong-hourly_" + etype + ".csv"
        # filepath = "data/armstrong/nov-13/armstrong-hourly_" + event_type + ".csv"
        filepath = "data/armstrong/gt_hourly_" + event_type + ".csv"
        # filepath = "data/armstrong/dec-07/armstrong_" + event_type + "_" + freq + ".csv"

    ts = pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                     squeeze=True)
    idx = pd.date_range(ts.index.min(), ts.index.max(), freq='H')
    ts_hourly = ts.reindex(idx, fill_value=0)
    return ts_hourly.groupby(pd.Grouper(freq=freq)).sum()


def load_knox_hourly(filepath=None, event_type="endpoint_malware",
                     freq="H"):
    if filepath is None:
        filepath = "data/knox/gt_hourly_" + event_type + ".csv"

    ts = pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                     squeeze=True)
    idx = pd.date_range(ts.index.min(), ts.index.max(), freq='H')
    ts_hourly = ts.reindex(idx, fill_value=0)
    return ts_hourly.groupby(pd.Grouper(freq=freq)).sum()


def load_armstrong_malicious_destination(filepath=None):
    if filepath is None:
        filepath = "./data/armstrong/gt_malicious-destination.csv"
    return pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                       squeeze=True)


def load_armstrong_malicious_email(filepath=None):
    if filepath is None:
        filepath = "./data/armstrong/gt_malicious-email.csv"
    return pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                       squeeze=True)


def load_knox_endpoint_malware(filepath=None):
    if filepath is None:
        filepath = "data/knox/gt_endpoint-malware.csv"
    ts = pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                     squeeze=True)
    idx = pd.date_range(ts.index.min(), ts.index.max())
    return ts.reindex(idx, fill_value=0)


def load_knox_malicious_destination(filepath=None):
    if filepath is None:
        filepath = "data/knox/gt_malicious-destination.csv"
    ts = pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                     squeeze=True)
    idx = pd.date_range(ts.index.min(), ts.index.max())
    return ts.reindex(idx, fill_value=0)


def load_knox_malicious_email(filepath=None):
    if filepath is None:
        filepath = "data/knox/gt_malicious-email.csv"
    ts = pd.read_csv(filepath, header=0, index_col=0, parse_dates=True,
                     squeeze=True)
    idx = pd.date_range(ts.index.min(), ts.index.max())
    return ts.reindex(idx, fill_value=0)


def load_armstrong_raw(filepath=None, event_type="endpoint-malware"):
    if filepath is None:
        filepath = "/Users/tozammel/Documents/projects/cause/data/armstrong-r3-oct-10/armstrong-setcore-20171009.json"

    data = []
    with open(filepath, 'r') as fh:
        entries = json.load(fh)
        print("Total number of all event types =", len(entries))
        for event in entries:
            if event['ground_truth']['event_type'] == event_type:
                data.append(
                    (pd.to_datetime(event['ground_truth']['occurred']),
                     pd.to_datetime(event['ground_truth']['reported'])
                     )
                )  # hourly model

    print("total number of events =", len(data))
    df = pd.DataFrame(data, columns=["date", "reported_date"])
    # df.set_index('date', inplace=True)
    print(df.shape)

    ts = df.groupby('date').size()
    # ts = df.groupby(['date', pd.Grouper(freq='6H')]).size()
    # print(ts)
    # exit(1)

    # ts = df.groupby(['date', df['date'].dt.hour]).count()
    # ts = df.groupby(pd.Grouper(key='date', freq="6H")).size()

    # ts = df.groupby('date').resample("6H", "count")

    # print(ts.head())
    # print(ts.index.min())
    # print(ts.index.max())
    # idx = pd.date_range(ts.index.min(), ts.index.max())
    # ts = ts.reindex(idx, fill_value=0)
    ts.name = 'count'
    ts.index.name = 'date'
    # print(ts.head())
    return ts
