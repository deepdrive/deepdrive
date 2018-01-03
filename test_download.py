from clint.textui import progress
import requests

url = 'https://d1y4edi1yk5yok.cloudfront.net/sim/deepdrive-sim-win-2.0.20171206173312.zip'

r = requests.get(url, stream=True)
path = 'asdf.zip'
with open(path, 'wb') as f:
    total_length = int(r.headers.get('content-length'))
    for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
        if chunk:
            f.write(chunk)
            f.flush()


