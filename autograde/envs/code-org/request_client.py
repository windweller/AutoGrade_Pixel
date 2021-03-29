import json
import requests
from requests_futures import sessions


# async def get(url, session):
#     # async with aiohttp.ClientSession() as session:
#     async with session.get(url) as response:
#         return response
#
#
# coroutines = [get("http://example.com") for _ in range(8)]
#
# results = loop.run_until_complete(asyncio.gather(*coroutines))

class Game:
    def __init__(self, port):
        self.port = port
        self.session = sessions.FuturesSession()

    def init(self, js_file, data_format):
        future = self._send("init", {"script": js_file, "format": data_format})
        result = future.result().status_code

        return result

    def _send(self, type, data):
        future = self.session.get(f'http://localhost:{self.port}/{type}/2?process=1&script=abc&format=img|state')
        return future


if __name__ == '__main__':
    g = Game(3300)
    # data_format is: "img" or "state" or "img|state" or "state|img"
    resp = g.init("var score = 0;", "img|state")

    print(resp)
