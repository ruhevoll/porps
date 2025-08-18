from polygon import RESTClient
import config
import json
from typing import cast
from urllib3 import HTTPResponse


client = RESTClient(api_key=config.API_KEY)

aggs = cast(
    HTTPResponse,
    client.get_aggs(
        'AAPL',
        1,
        'day',
        '2025-07-01',
        '2025-07-31',
        raw = True
    ),
)

data = json.loads(aggs.data)
print(data)