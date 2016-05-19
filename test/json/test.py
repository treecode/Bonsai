import json
from pprint import pprint

with open('release.json') as data_file:    
    data = json.load(data_file)

pprint(data)

print data["task"]
