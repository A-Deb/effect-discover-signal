#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


class ModelResults:
    def __init__(self, model_name, estimated_param):
        self.name = model_name
        self.learned_param = estimated_param

