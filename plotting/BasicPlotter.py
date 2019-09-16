from abc import ABCMeta
import sys
import json
from datetime import date, datetime, timedelta as td
import matplotlib.pyplot as plt

#sys.path.append('../basic_source_code/')
#
#from MongoDbClient import MongoDbClient

class BasicPlotter(metaclass=ABCMeta):
    def __init__(self, start_date, end_date):
        plt.rcParams.update({"xtick.labelsize": 25.0,\
                             "ytick.labelsize": 25.0,\
                             "legend.fontsize": 25.0})
        self.start_date = start_date
        self.end_date = end_date
        
        self.ground_truth, self.likely_domains = self.get_data_from_file()

#        self.colors = ["#f7f7f7", "#cccccc", "#969696", "#636363", "#252525"]
        self.colors = [(1.0, 0.4980392156862745, 0.054901960784313725),\
                       (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),\
                       (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),\
                       (0.5803921568627451, 0.403921568627451, 0.7411764705882353),\
                       (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),\
                       (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),\
                       (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),\
                       (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),\
                       (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),\
                       (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]
        self.patterns = ("////", "\\\\\\\\", "----", "++++", "....", "****", "oooo", "0000", "====", "::::")

    def get_data_from_file(self):
        ground_truth = []
        with open("../backups/ground_truth.json", "r") as f:
            for line in f:
                ground_truth.append(json.loads(line))
        likely_domains = []
        with open("../backups/likely_domains.json", "r") as f:
            for line in f:
                likely_domains.append(json.loads(line))

        return ground_truth, likely_domains

#    def get_data_from_db(self):
#        mongo_db_client = MongoDbClient()
#        ground_truth = self.mongo_db_client.get_all("ground_truth")
#        likely_domains = self.mongo_db_client.get_all("likely_domain")
#
#        return ground_truth, likely_domains

    def initialize_date_dict(self):
        date_dict = {}
        for i in range((self.end_date - self.start_date).days + 1):
            date_dict[str(self.start_date + td(days=i))] = 0

        return date_dict

    def prepare_dates_values(self, data_dict):
        dates = [datetime.strptime(key, "%Y-%m-%d").date() for key in data_dict.keys()]
        values = [int(value) for value in data_dict.values()]

        return dates, values

    def add_important_events(self, ax, start_y=85.0, stepsize=5.0):
        fontsize_value = 25.0
        ax.text(x=datetime.strptime("2016-08-05", "%Y-%m-%d").date(), y=start_y, s="2", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-08-04", "%Y-%m-%d").date(), color="k")
#        ax.text(x=datetime.strptime("2016-08-17", "%Y-%m-%d").date(), y=start_y-stepsize, s="Check Point Decryptor", fontsize=fontsize_value)
#        ax.axvline(x=datetime.strptime("2016-08-16", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-08-31", "%Y-%m-%d").date(), y=start_y, s="3", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-08-30", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-10-05", "%Y-%m-%d").date(), y=start_y, s="4(X)", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-10-04", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-10-20", "%Y-%m-%d").date(), y=start_y, s="4.0.2", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-10-19", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-10-28", "%Y-%m-%d").date(), y=start_y-stepsize, s="4.1.0", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-10-27", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-11-01", "%Y-%m-%d").date(), y=start_y-2*stepsize, s="4.1.1", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-10-31", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-11-08", "%Y-%m-%d").date(), y=start_y-3*stepsize, s="4.1.3", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-11-07", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-11-09", "%Y-%m-%d").date(), y=start_y-4*stepsize, s="4.1.4", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-11-08", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-11-10", "%Y-%m-%d").date(), y=start_y, s="4.1.5", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-11-09", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-11-23", "%Y-%m-%d").date(), y=start_y-stepsize, s="4.1.6", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-11-22", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-11-24", "%Y-%m-%d").date(), y=start_y-2*stepsize, s="5.0.0", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-11-23", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-11-26", "%Y-%m-%d").date(), y=start_y-3*stepsize, s="5.0.1", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-11-25", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2016-12-02", "%Y-%m-%d").date(), y=start_y, s="Red", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2016-12-01", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2017-01-04", "%Y-%m-%d").date(), y=start_y, s="Red 1.1", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2017-01-03", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2017-01-26", "%Y-%m-%d").date(), y=start_y, s="Red 1.2", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2017-01-25", "%Y-%m-%d").date(), color="k")
        ax.text(x=datetime.strptime("2017-03-31", "%Y-%m-%d").date(), y=start_y, s="6", fontsize=fontsize_value)
        ax.axvline(x=datetime.strptime("2017-03-30", "%Y-%m-%d").date(), color="k")