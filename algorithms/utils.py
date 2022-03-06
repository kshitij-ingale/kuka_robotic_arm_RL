import yaml

class OD(dict):
    def __init__(self, dict_):
        for key, value in dict_.items():
            if isinstance(value, (list, tuple)):
                # Check if dict in list/tuple and convert to OD
                value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
            elif isinstance(value, dict) and not isinstance(value, self.__class__):
                # Convert nested dict to OD
                value = self.__class__(value)
            setattr(self, key, value)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        super().__setitem__(key, value)

    __setitem__ = __setattr__

def flatten(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def parse_config(file_path):
    with open(file_path, 'r') as f:
        parameters = yaml.safe_load(f)
    return OD(parameters)
