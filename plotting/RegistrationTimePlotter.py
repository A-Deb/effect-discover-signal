import sys
import logging
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from BasicPlotter import BasicPlotter

class RegistrationTimePlotter(BasicPlotter):
    def __init__(self, start_date, end_date):
        BasicPlotter.__init__(self, start_date, end_date)

        self.registration_dates_gt, self.registration_values_gt = self.prepare_data_registration(self.ground_truth)
        self.registration_dates_likely, self.registration_values_likely = self.prepare_data_registration(self.likely_domains)
        self.detection_dates_gt, self.detection_values_gt = self.prepare_data_detection(self.ground_truth)

    def prepare_data_registration(self, domain_data):
        """Collect all registration dates and count number of registered domains per date
        """
        registration_dict = self.initialize_date_dict()
        for elem in domain_data:
            if "whois" in elem and "registered" in elem["whois"]:
                registration_date = elem["whois"]["registered"].split("T")[0]
                if registration_date != "N/A" and registration_date > "2016-05-31":
                    registration_dict[registration_date] += 1

        return self.prepare_dates_values(registration_dict)

    def prepare_data_detection(self, domain_data):
        """Collect all detection dates and count number of detected domains per date
        """
        detection_dict = self.initialize_date_dict()
        for elem in domain_data:
            if "date_added" in elem:
                detection_date = elem["date_added"].split(" ")[0]
                if detection_date != "" and detection_date > "2016-05-31":
                    detection_dict[detection_date] += 1

        return self.prepare_dates_values(detection_dict)

    def plot_registration_dates(self):
        """Plot registration dates of both ground truth and likely domains
        """
        fig, ax = self.prepare_plot(75.0, 5.0, "Months", "# registered domains")
        gt_bar = ax.bar(self.registration_dates_gt, self.registration_values_gt, 1, color=self.colors[0], alpha=0.8)
        likely_bar = ax.bar(self.registration_dates_likely, self.registration_values_likely, 1, bottom=self.registration_values_gt, color=self.colors[4], alpha=0.8)
        ax.legend((gt_bar[0], likely_bar[0]), ("Known Cerber domains", "Cerber candidates"), loc=2)
        plt.show()

    def plot_detection_dates(self):
        """Plot detection dates of ground truth
        """
        fig, ax = self.prepare_plot(28.0, 2.0, "Months", "# detected domains")
        gt_bar = ax.bar(self.detection_dates_gt, self.detection_values_gt, 1, color=self.colors[0], alpha=0.8)
        ax.legend([gt_bar[0]], ["Known Cerber domains"], loc=2)
        plt.show()

    def plot_detection_registration_dates(self):
        """Plot registration and detection dates of ground truth 2,9
        """
        fig, ax = self.prepare_plot(75.0, 5.0, "Months", "# registered/detected domains")
        registration_bar = ax.bar(self.registration_dates_gt, self.registration_values_gt, 1, color=self.colors[2], alpha=0.8)
        detection_bar = ax.bar(self.detection_dates_gt, self.detection_values_gt, 1, bottom=self.registration_values_gt, color=self.colors[9], alpha=0.8)
        ax.legend([registration_bar[0], detection_bar[0]], ["Registration", "Detection"], loc=1)
        plt.show()

    def prepare_plot(self, important_events_x, important_events_y, label_x, label_y):
        """Prepare plot environment, i.e., set labels, formatter etc.
        """
        fig, ax = plt.subplots(figsize=(20, 3))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_xlim(self.start_date, self.end_date)
        ax.yaxis.grid()
        self.add_important_events(ax, important_events_x, important_events_y)
        plt.xlabel(label_x, labelpad=30, fontsize=25.0)
        plt.ylabel(label_y, labelpad=30, fontsize=25.0)
        ax.tick_params(axis="x", pad=10)
        ax.tick_params(axis="y", pad=10)
        fig.autofmt_xdate()

        return fig, ax

if __name__ == "__main__":
    rtp = RegistrationTimePlotter(date(2016, 6, 1), date(2017, 6, 1))
    #rtp.plot_registration_dates()
    #rtp.plot_detection_dates()
    rtp.plot_detection_registration_dates()