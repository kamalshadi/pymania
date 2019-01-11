import json
import os
import pymania


# only one configuration instance allowed at a time (singletone)
class Config:
    class __Config:
        def __init__(self, **kwargs):
            for w in kwargs:
                self.__setattr__(w,kwargs[w])
    instance = None
    def __init__(self, **kwargs):
        if not Config.instance:
            Config.instance = Config.__Config(**kwargs)
        else:
            for w in kwargs:
                Config.instance.__setattr__(w,kwargs[w])
    def __getattr__(self, name):
        return getattr(self.instance, name)


fp = os.path.join(os.path.dirname(pymania.__file__),'default_config.json')
with open(fp) as f:
    config = Config(**json.load(f))
