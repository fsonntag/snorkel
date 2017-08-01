
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def download(url, outfname):
    """
    Download target URL

    :param url:
    :param outfname:
    :return:
    """
    try:
        data = urlopen(url)
        with open(outfname, "wb") as f:
            f.write(data.read())
    except HTTPError as e:
        print("HTTP Error:", e.code, url)
    except URLError as e:
        print("URL Error:", e.reason, url)