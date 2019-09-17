"""Metrics module."""
# pylint: disable=deprecated-lambda,invalid-name,no-else-return,no-self-use,too-many-public-methods
from datetime import datetime, timedelta
import math
import re

import numpy as np
import tldextract as tlde

TIME_PATTERN1 = "%Y-%m-%dT%H:%M:%S.%fZ"
TIME_PATTERN2 = "%Y-%m-%dT%H:%M:%SZ"


def parse_time(time_string):
    try:
        result = datetime.strptime(time_string, TIME_PATTERN1)
    except:
        result = datetime.strptime(time_string, TIME_PATTERN2)
    return result

class Metrics(object):
    """Metrics class."""

    @staticmethod
    def split_or_none(fld):
        """Return warning or ground truth field split by ' ' or None if field is None
        :param fld: warning or ground truth field
        :type fld: str or None
        :rtype: [str] or None
        :returns: list of strings that result from splitting fld or None if field is None
        """
        if fld is not None:
            split_fld = fld.split(' ')
        else:
            split_fld = None

        return split_fld

    def num_microseconds(self, td):
        """Return number of microseconds.

        :param td: timestamp to be converted to microseconds
        :type td: timestamp
        :rtype: float
        :returns: time in microseconds
        """
        return float(td.microseconds + 1000000 * (td.seconds + 86400 * td.days))

    def sim_time(self, a, b, thresh):
        r"""Return similarity of timestamps.

        ..math::
            sim_{time}(w,g) = 1 - \min(1, |t_w-t_g|/t_{thres})

        :param a: time of gt occurrence
        :type A: datetime
        :param B: time of warning predicted occurrence
        :type B: datetime
        :param thresh: maximum difference allowed for a match in days
        :type thresh: time delta
        :rtype: float
        :returns: time similarity metric
        """
        num = self.num_microseconds(abs(a - b))
        thresh = self.num_microseconds(thresh)
        score = 1 - min(1, num / thresh)
        return score

    def ensure_list(self, x):
        """Ensure parameter is a list."""
        return x if isinstance(x, list) else [x]

    def multi_score(self, r, s, f):
        """Apply function to powerset of two lists."""
        return [None] if r is None else [0.0] if s is None else [f(x, y)
                                                                 for y in self.ensure_list(s)
                                                                 for x in self.ensure_list(r)]

    def n_score(self, a, denom=None):
        """Return score of nested conditions.

        :param a: linked list of conditions evaluated hierarchically
        :type a: linked list
        :param denom: the normalizing constant (default: None)
        :type denom: float | None
        """
        def depth(x):  # pylint: disable=missing-docstring
            return 1.0 + depth(x[1]) if isinstance(x[1], tuple) else float(len(x))

        def isSomething(x):  # pylint: disable=missing-docstring
            return x is not None and x is not False and x != 0.0

        if denom is None:
            if isinstance(a, tuple):
                denom = depth(a)
            elif not isinstance(a, tuple):
                denom = 1.0

        if isinstance(a, tuple):
            if all([bool(x) for x in a]):
                return float(isSomething(a[0])) * (1 / denom) + self.n_score(a[1], denom=denom)
            else:
                return float(isSomething(a[0])) * (1 / denom)

        elif not isinstance(a, tuple) and a != 0.0 and a:
            if a is True:
                return float(a is not None) / denom
            else:
                return a / denom
        else:
            return 0.0

    def base_criteria(self, g, w, thr1=0, thr2=7):
        """Determine if match passes base criteria.

        :param g: ground truth event
        :type g: MetricGroundTruth
        :param w: warning sumission
        :type w: MetricWarning
        :param thr1: threshold on first condition in seconds
        :type thr1: int
        :param thr2: threshold on second condition in days
        :type thr2: int
        """


        at_least_thresh1_after = (parse_time(g.reported) -
                                  parse_time(w.submitted)).total_seconds() >= thr1
        no_more_than_thresh2_days = abs(parse_time(w.occurred) -
                                        parse_time(g.occurred)).total_seconds() / 86400 < thr2
        match_event_type = g.event_type == w.event_type
        match_target_industry = g.target_industry == w.target_industry
        return at_least_thresh1_after and no_more_than_thresh2_days and match_event_type and match_target_industry

    def indicator(self, a, b):
        """Indicate equality of two objects.

        :param a: item to compare to b
        :type a: anything that can be compared
        :param b: item to compare to a
        :type b: anything that can be compared
        :returns: 1 if a == b, 0 othewise
        """
        return self.n_score(a == b)

    def lev_dis(self, a, b):
        r"""Return Levenshtein distancse atwixt two strings.

        ..math::
            \displaystyle lev_{a,b}(i,j) = \begin{cases} \max(i,j) & \text{if } \min(i,j)
            = 0\\ \min \begin{cases} lev_{a,b}(i-1,j) + 1\\ lev_{a,b}(i,j-1) + 1\\lev_{a,b}(i-1,j-1)
            + \mathbf{1}_{a_i \ne b_j}\end{cases} \end{cases}

        :param a: a string
        :type a: str
        :param b: a string
        :type b: str
        :rtype: float
        :returns: the levenshtein distance between string a and string b
        """
        l_mat = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]

        l_mat = np.matrix(l_mat)

        for i in range(1, len(a) + 1):
            l_mat[i, 0] = i

        for j in range(1, len(b) + 1):
            l_mat[0, j] = j

        for j in range(1, len(b) + 1):
            for i in range(1, len(a) + 1):
                l_mat[i, j] = min(l_mat.item((i - 1, j)) + 1, l_mat.item((i, j - 1)) + 1,
                                  l_mat.item((i - 1, j - 1)) + int(a[i - 1] != b[j - 1]))

        return l_mat.item((len(a), len(b)))

    def sim_ld(self, a, b, alpha=2, beta=100):
        r"""Return similarity of two strings.

        ..math::
            \displaystyle sim_{LD}(s_1, s_2) = \exp\left(-\beta\left(\frac{LD(s_1, s_2)}
            {length(s_1)\cdot length(s_2)}\right)^\alpha\right)

        :param a: the first item (ground truth) to be compared
        :type a: str
        :param b: the second item (warning) to be compared
        :type b: str
        :param alpha: first tuning parameter (default: 2)
        :type alpha: float
        :param beta: second tuning parameter (default: 100)
        :type beta: float
        :rtype: float
        :returns: similarity measure for the two strings
        """
        if a is None:
            return None
        if b is None:
            return 0.0

        if a == b:
            return 1.0

        la, lb = len(a), len(b)
        ld = float(self.lev_dis(a, b))
        return np.exp(-beta * math.pow((ld / (la + lb)), alpha))

    def sim_recip(self, a, b):
        """Return similarity of two recipients.

        :param a: a recipient of malicious email in ground truth
        :type a: str
        :param b: a recipient of malicious email in warning
        :type b: str
        """
        return self.sim_ld(a, b, alpha=2.25, beta=80)

    def sim_sender(self, a, b):
        """Return similarity of two senders.

        :param a: the sender of malicious email in ground truth
        :type a: str
        :param b: the senders of malicious email in warning
        :type b: str
        """
        return self.sim_ld(a, b, alpha=4, beta=100)

    def jacc(self, A, B):
        r"""Return Jaccard similarity of two sets.

        ..math::
            J(A,B) = \displaystyle \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}

        :param A: list of items
        :type A: list
        :param B: list of items
        :type B: list
        :rtype: float
        :returns: jaccard similarity score
        """
        if not isinstance(A, list):
            A = [A]
        if not isinstance(B, list):
            B = [B]
        inter = float(len(list(filter(lambda x: x in A, B))))
        denom = len(A) + len(B) - inter
        return inter / denom

    def sim_hash(self, h1, h2):
        """Return hash similarity.

        :param h1: first hash from warning
        :type h1: str
        :param h2: second hash from ground truth
        :type h2: str
        :rtype: int
        :returns: hash similarity score
        """
        if h1 is None:
            return None
        if h2 is None:
            return 0.0

        n = min(len(h1), len(h2))
        return int(h1[:n] == h2[:n])

    def sim_ip(self, ip1, ip2):
        """Return IP similarity.

        :param ip1: ip address from warning
        :type ip1: IPAddress
        :param ip2: ip address from ground truth
        :type ip2: IPAddress
        :rtype: float
        :returns: similarity of two IP addresses
        """
        if ip1 is None:
            return None
        if ip2 is None:
            return 0.0

        def decay(x, y):  # pylint: disable=missing-docstring
            return np.exp(-0.04 * abs(x - y))

        def ip_mat(x, y, z):  # pylint: disable=missing-docstring
            return '.'.join(x.split('.')[:z]) == '.'.join(y.split('.')[:z])

        ipsp1 = [int(x) for x in ip1.split('.')]
        ipsp2 = [int(x) for x in ip2.split('.')]

        ip_sim = 0.0
        if (ip1 == ip2):
            ip_sim = 1.0
        elif ip_mat(ip1, ip2, 3):
            ip_sim = 0.9 + 0.1 * decay(ipsp1[3], ipsp2[3])
        elif ip_mat(ip1, ip2, 2):
            ip_sim = 0.6 + 0.3 * decay(ipsp1[2], ipsp2[2])
        elif ip_mat(ip1, ip2, 1):
            ip_sim = 0.15 + 0.45 * decay(ipsp1[1], ipsp2[1])
        return ip_sim

    def sim_path(self, p1, p2):
        """Return similarity of two paths.

        :param p1: path from ground truth
        :type p1: string
        :param p2: path from warning
        :type p2: string
        :rtype: float
        :returns: similarity of two paths
        """
        if p1 is None:
            return None
        if p2 is None:
            return 0.0

        if p1 == p2:
            return 1.0

        def trim_slash(x):  # pylint: disable=missing-docstring
            if x == '':
                return x
            if x[0] == '/' and x[-1] == '/':
                return x[1:-1]
            elif x[0] == '/':
                return x[1:]
            elif x[-1] == '/':
                return x[:-1]
            else:
                return x

        # This is neccessary for dealing with PC paths
        def smart_split(x, d1, d2):  # pylint: disable=missing-docstring
            is_1 = len(x.split(d1)) > 1
            is_2 = len(x.split(d2)) > 1

            if x == '' or not (is_1 or is_2):
                return [x]
            if is_1:
                return x.split(d1)
            if is_2:
                x = x.replace('c:\\', '')
                x = x.replace('C:\\', '')
                return x.split(d2)

        psp1 = smart_split(trim_slash(p1), '/', '\\')
        psp2 = smart_split(trim_slash(p2), '/', '\\')

        if len(psp1) == 1 and len(psp2) == 1:
            return self.sim_ld(p1, p2)

        return self.jacc(psp1, psp2)

    def split_addr(self, a):
        """Split URL into requisite components.

        :param a: address that needs to be split into groups accoring to G.7 on p. 42 of GTHBv2.0
        :type a: str
        """
        a = a.replace('http://', '')
        a = a.replace('https://', '')

        addr = tlde.extract(a)
        is_ip = tlde.tldextract.looks_like_ip(addr.domain)
        if is_ip:
            ip = addr.domain
            path_and_params = a[a.index(ip)+len(ip):].split('?')
            path = path_and_params[0]
            if len(path_and_params) > 1:
                params = path_and_params[1:]
            else:
                params = ''
            return {'ip': ip, 't3': None, 't2': None, 'path': path, 'params': params, 'url/ip': 'ip'}
        else:
            t3 = addr.subdomain
            t2 = addr.registered_domain
            path_and_params = a[a.index(addr.fqdn)+len(addr.fqdn):].split('?')
            path = path_and_params[0]
            if len(path_and_params) > 1:
                params = path_and_params[1:]
            else:
                params = ''
            return {'t3': t3, 't2': t2, 'ip': None, 'path': path, 'params': params, 'url/ip': 'url'}

    def sim_address(self, a, b):
        """Return similarity of URLs.

        :param a: address (url or ip) from ground truth
        :type a: str
        :param b: address (url or ip) from warning
        :type b: str
        """
        if a is None:
            return None
        if b is None:
            return 0.0

        parts_a = self.split_addr(a)
        parts_b = self.split_addr(b)

        def makeAplusB(self, p_a, p_b):  # pylint: disable=missing-docstring
            if p_a['url/ip'] == p_b['url/ip']:
                if p_a['url/ip'] == 'url':
                    A = 0.4 * self.indicator(p_a['t2'], p_b['t2'])
                    B = 0.1 * self.jacc(p_a['t3'].split('.'), p_b['t3'].split('.'))
                    return A + B
                else:
                    return 0.5 * self.sim_ip(p_a['ip'], p_b['ip'])
            else:
                return 0.0

        AplusB = makeAplusB(self, parts_a, parts_b)
        C = 0.3 * self.sim_path(parts_a['path'], parts_b['path'])
        D = 0.2 * self.sim_ld(parts_a['params'], parts_b['params'], beta=10)
        return AplusB + C + D

    def event_type_score(self, g, w):
        """Return score of event types.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        if w.event_type == "malicious-email":
            return self.n_score((w.event_type == g.event_type, w.event_subtype == g.event_subtype))
        else:
            return self.indicator(w.event_type, g.event_type)

    def occ_time_score(self, g, w, thresh=7):
        r"""Return occurence timestamp score.

        ..math::
            sim_{time}(w,g) = 1 - \min(1, |t_w-t_g|/t_{thres})

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        :param thresh: time threshold (in days)
        :type thresh: float
        """
        t1 = parse_time(g.occurred)
        t2 = parse_time(w.occurred)

        return self.sim_time(t1, t2, timedelta(days=thresh))

    def target_score(self, g, w):
        """Return target entity score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        A = bool(max(self.multi_score(g.target_industry, w.target_industry, self.indicator)))
        B = bool(max(self.multi_score(g.target_organization, w.target_organization, self.indicator)))
        if g.event_type == 'malicious-email':
            C = max(self.multi_score(g.target_entity, w.target_entity, self.sim_recip))
        else:
            C = max(self.multi_score(g.target_entity, w.target_entity, self.sim_ld))

        if C is None:
            return self.n_score((A, B))
        else:
            return self.n_score((A, (B, C)))

    def event_details_score_all(self, g, w):
        """Return score of all event details.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        functions = {
            'malicious-email': self.event_details_score_mal_email,
            'malicious-destination': self.event_details_score_mal_dest,
            'endpoint-malware': self.event_details_score_moe
        }

        return functions[w.event_type](g, w)

    def event_details_score_mal_email(self, g, w):
        """Return malicious email score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        sen = self.sender_score(g, w)
        sub = self.subject_score(g, w)
        lin = self.link_score(g, w)
        att = self.attachments_score(g, w)

        try: scores = list(filter(lambda x: x is not None, [sen, sub, max(lin, att)]))
            
        except TypeError:
            scores=0
        return np.mean(scores)

    def event_details_score_mal_dest(self, g, w):
        """Return malicious destination score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        thr = self.threat_designation_score(g, w)
        m_url = self.mal_url_score(g, w)
        m_ip = self.mal_ip_score(g, w)

        addr_scores = list(filter(lambda x: x is not None, [m_url, m_ip]))

        if not addr_scores:
            return thr
        elif thr is None:
            max(addr_scores)
        else:
            coeff = 1 / 2.0
            return coeff * thr + coeff * (max(addr_scores))

    def event_details_score_moe(self, g, w):
        """Return malware on endpoint score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        thr = self.threat_designation_score(g, w)
        fln = self.filename_score(g, w)
        pth = self.path_score(g, w)
        hsh = self.hash_score(g, w)

        file_scores = [fln, pth, hsh]
        file_scores = list(filter(lambda x: x is not None, file_scores))

        if not file_scores:
            return thr
        elif thr is None:
            coeffs = [0.5, 1 / float(len(file_scores))]
            return coeffs[1] * (sum(file_scores))
        else:
            coeffs = [0.5, 1 / float(2.0 * len(file_scores))]
            return coeffs[0] * thr + coeffs[1] * (sum(file_scores))

    def subject_score(self, g, w):
        """Return subject score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        scores = self.multi_score(g.email_subject, w.email_subject, self.sim_ld)

        return max(scores)

    def sender_score(self, g, w):
        """Return sender score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        email_sender = self.split_or_none(g.email_sender)

        scores = self.multi_score(email_sender, w.email_sender, self.sim_sender)

        return max(scores)

    def link_score(self, g, w):
        """Return link score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        thr = self.threat_designation_score(g, w)
        m_url = self.mal_url_score(g, w)
        m_ip = self.mal_ip_score(g, w)

        addr_scores = list(filter(lambda x: x is not None, [m_url, m_ip]))

        if not addr_scores:
            return thr
        elif thr is None:
            return max(addr_scores)
        else:
            coeff = 1 / 2.0
            return coeff * thr + coeff * max(addr_scores)

    def attachments_score(self, g, w):
        """Return attachments score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        thr = self.threat_designation_score(g, w)
        fln = self.filename_score(g, w)
        hsh = self.hash_score(g, w)

        file_scores = list(filter(lambda x: x is not None, [fln, hsh]))

        if not file_scores:
            return thr
        elif thr is None:
            return max(file_scores)
        else:
            coeff = 1 / 2.0
            return coeff * thr + coeff * max(file_scores)

    def threat_designation_score(self, g, w):
        """Return threat designation score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        threat_designation_type = self.split_or_none(g.threat_designation_type)
        threat_designation_family = self.split_or_none(g.threat_designation_family)
        detector_classification = self.split_or_none(g.detector_classification)

        a = bool(max(self.multi_score(threat_designation_type, w.threat_designation_type, self.indicator)))
        b = bool(max(self.multi_score(threat_designation_family, w.threat_designation_family, self.indicator)))
        c = bool(max(self.multi_score(detector_classification, w.detector_classification, self.indicator)))
        
        if g.threat_designation_type is None:
            return None
        elif g.threat_designation_family is None:
            return self.n_score(a)
        elif g.detector_classification is None:
            return self.n_score((a, b))
        else:
            return self.n_score((a, (b, c)))

    def mal_url_score(self, g, w):
        """Return malicious URL score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        if g.addresses_url and g.addresses_ip:
            addresses_url = g.addresses_url.split(' ')
            addresses_ip = g.addresses_ip.split(' ')
            url_and_ip = addresses_url + addresses_ip
        elif g.addresses_url and not g.addresses_ip:
            url_and_ip = g.addresses_url.split(' ')
        elif g.addresses_ip and not g.addresses_url:
            url_and_ip = g.addresses_ip.split(' ')
        else:
            url_and_ip = None

        scores = self.multi_score(url_and_ip, w.addresses_url, self.sim_address)

        return max(scores)

    def mal_ip_score(self, g, w):
        """Return malicious IP score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        if g.addresses_url and g.addresses_ip:
            addresses_url = g.addresses_url.split(' ')
            addresses_ip = g.addresses_ip.split(' ')
            url_and_ip = addresses_url + addresses_ip
        elif g.addresses_url and not g.addresses_ip:
            url_and_ip = g.addresses_url.split(' ')
        elif g.addresses_ip and not g.addresses_url:
            url_and_ip = g.addresses_ip.split(' ')
        else:
            url_and_ip = None

        scores = self.multi_score(url_and_ip, w.addresses_ip, self.sim_address)

        return max(scores)

    def filename_score(self, g, w):
        """Return filename score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        files_filename = self.split_or_none(g.files_filename)

        scores = self.multi_score(files_filename, w.files_filename, self.sim_ld)

        return max(scores)

    def path_score(self, g, w):
        """Return path score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        files_path = self.split_or_none(g.files_path)

        scores = self.multi_score(files_path, w.files_path, self.sim_path)

        return max(scores)

    def hash_score(self, g, w):
        """Return hash score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        g_files_hash = self.split_or_none(g.files_hash)
        w_files_hash = self.split_or_none(w.files_hash)

        scores = self.multi_score(g_files_hash, w_files_hash, self.sim_hash)

        return max(scores)

    def probability_score(self, m, w):
        """Return probability score.

        :param m: 1 if matched, 0 if unmatched
        :type m: int
        :param w: warning event
        :type w: MetricWarning
        """
        prob = float(w.probability)
        return 1 - (m - prob)**2

    def lead_time_score(self, g, w):
        """Return lead time score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        rep = parse_time(g.reported)
        iss = parse_time(w.submitted)
        diff = rep - iss
        return diff.total_seconds() / 86400

    def utility_time_score(self, g, w):
        """Return utility time score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        occ = parse_time(g.occurred)
        iss = parse_time(w.submitted)
        diff = occ - iss
        return diff.total_seconds() / 86400

    def quality_score(self, g, w):
        """Return quality score.

        :param w: warning event
        :type w: MetricWarning
        :param g: ground truth event
        :type g: MetricGroundTruth
        """
        evt = self.event_type_score(g, w)
        occ = self.occ_time_score(g, w)
        tar = self.target_score(g, w)
        evd = self.event_details_score_all(g, w)

        scores = [evt, occ, tar, evd]
        scores = list(filter(lambda x: x is not None and not np.isnan(x), scores))

        return np.mean(scores) * 4.0