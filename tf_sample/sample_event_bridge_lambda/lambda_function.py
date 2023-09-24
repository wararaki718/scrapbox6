import json


def lambda_handler(event, context) -> dict:
    print(json.dumps(event))
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Hello from Lambda!"
        }),
    }
