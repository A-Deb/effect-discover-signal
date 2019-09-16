#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import subprocess
import os
from elasticsearch import serializer

class Elasticsearch(object):
    
    def __init__(self, hosts, http_auth=None, port=80):
        print("importing ATL es-curl")
        self.hosts = hosts
        self.http_auth = http_auth
        self.port = port
    
    def search(self, index="", body=None, params=None):
        query = body
        os.environ['http_proxy'] = 'http://proxy.atl.lmco.com:3128'
        print (index)
        data = json.dumps(query)
        #req = "curl --silent -u " + self.http_auth[0] + ":" + self.http_auth[1] + " -H 'Content-Type: application/json' -d '" + data  + "' " + self.hosts[0] + index + "/_search"
        req = "curl --silent -u " + self.http_auth[0] + ":" + self.http_auth[1] + " -H 'Content-Type: application/json' -d '" + data  + "' " + self.hosts[0] + index + "/_search?preference=effect"
        if params is not None:
            req = req + "&" + params
        
        print(req)
        out = subprocess.check_output(req, shell=True).decode("utf-8")
        
        default_mimetype='application/json'
        serial=serializer.JSONSerializer()
        _serializers = serializer.DEFAULT_SERIALIZERS.copy()
        deserializer = serializer.Deserializer(_serializers, default_mimetype)
        result = deserializer.loads(out)
        return result

