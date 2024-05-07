import time

import torch
from tqdm import tqdm


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        t_min = (end_time - start_time) / 60
        print(f"/n{func.__name__} executed in {t_min:.4f} min.")
        return result

    return wrapper


def batch_iterate(lst, batch_size):
    for i in tqdm(range(0, len(lst), batch_size)):
        yield lst[i : i + batch_size]


def val_to_label(val):
    if val == "positive":
        return 1
    if val == "negative":
        return 0
    else:
        return -1


def label_to_val(label):
    if label == 1:
        return "positive"
    if label == 0:
        return "negative"
    else:
        return "unknown"


def get_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device
