# VoiceBatch Studio Pro - Utilities
class AttrDict(dict):
    """
    यह क्लास डिक्शनरी (dict) को ऑब्जेक्ट की तरह इस्तेमाल करने देती है।
    इससे कोड में config['sample_rate'] की जगह config.sample_rate लिख सकते हैं।
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
