def _get_keys(dicts):
    keys = [set(dict_.keys()) for dict_ in dicts]
    assert all([i == j for i, j in zip(keys[1:], keys[:-1])]), keys

    return keys[0]


def dict_mean(dicts):
    keys = _get_keys(dicts)
    result = {}
    for k in keys:
        result[k] = sum([dict_[k] for dict_ in dicts]) / len(dicts)
    return result


def dict_sum(dicts):
    keys = _get_keys(dicts)
    result = {}
    for k in keys:
        result[k] = sum([dict_[k] for dict_ in dicts])
    return result


def dict_max(dicts):
    keys = _get_keys(dicts)
    result = {}
    for k in keys:
        result[k] = max([dict_[k] for dict_ in dicts])
    return result


def dict_min(dicts):
    keys = _get_keys(dicts)
    result = {}
    for k in keys:
        result[k] = min([dict_[k] for dict_ in dicts])
    return result
