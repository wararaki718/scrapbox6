import datetime


def datediff(days: int) -> str:
    now = datetime.datetime.now() - datetime.timedelta(days=days)
    return now.strftime("%Y-%m-%d")
