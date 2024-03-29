import json
import pickle

def read_json(file):
    with open(file, "r") as f:
        lines = [json.loads(line) for line in f]
    if len(lines) == 1:
        return lines[0]
    return lines


def write_json(data, file):
    if not isinstance(data, list):
        data = [data]
    with open(file, "w+") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def read_pickle(file):
    _df = pickle.load(open(file, "rb"))
    return _df


def write_pickle(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f, protocol=4)