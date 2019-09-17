import numpy as np
import math

def lev_dis(a, b):
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

def sim_ld(a, b, alpha=2, beta=100):
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
    ld = float(lev_dis(a, b))
    return np.exp(-beta * math.pow((ld / (la + lb)), alpha))

def sim_hash(h1, h2):
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

def sim_ip(ip1, ip2):
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

def sim_path(p1, p2):
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
        return sim_ld(p1, p2)

    return jacc(psp1, psp2)

def split_addr(a):
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



#print(sim_path('b','b'))
print(sim_ld('aaaaaakkkkkkkkkkkk','b'))
