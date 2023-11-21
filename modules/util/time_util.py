from datetime import datetime


def get_string_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
