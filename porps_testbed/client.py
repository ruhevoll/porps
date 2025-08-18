from polygon import RESTClient
import config
import json
from typing import cast
from urllib3 import HTTPResponse
from plotly import graph_objects as go
import pandas as pd
import numpy as np 
import talib

client = RESTClient(api_key=config.API_KEY)

aggs = cast(
    HTTPResponse,
    client.get_aggs(
        'AAPL',
        1,
        'day',
        '2025-01-01',
        '2025-07-31',
        raw = True,
        limit = 50000
    ),
)

data = json.loads(aggs.data)
# print(data)

for item in data:
    if item == 'results':
        rawData = data[item]
closeList = []
openList = []
highList = []
lowList = []
timeList = []

for bar in rawData:
    for category in bar:
        if category == 'c':
            closeList.append(bar[category])
        elif category == 'o':
            openList.append(bar[category])
        elif category == 'h':
            highList.append(bar[category])
        elif category == 'l':
            lowList.append(bar[category])
        elif category == 't':
            timeList.append(bar[category]) 

closeList = np.array(closeList)
upper, middle, lower = talib.BBANDS(closeList, timeperiod=20, nbdevdn=2, matype = 0)
times = []
for time in timeList:
    times.append(pd.Timestamp(time, tz='EST', unit='ms'))

fig = go.Figure()
fig.add_trace(go.Candlestick(x=times, open=openList, high = highList, low = lowList, close = closeList, name = 'AAPL market data for July 2025'))
fig.add_trace(go.Scatter(x = times, y = upper, line=dict(color='blue'), name='BB Upper'))
fig.add_trace(go.Scatter(x = times, y = lower, line=dict(color='lightblue'), name='BB Lower'))
fig.add_trace(go.Scatter(x = times, y = middle, line=dict(color='red'), name='BB Middle'))
fig.show()