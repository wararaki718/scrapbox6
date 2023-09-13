from datetime import datetime, timedelta


def datediff(days: int) -> str:
    now = datetime.now() - timedelta(days=days)
    return now.strftime("%Y-%m-%d")
