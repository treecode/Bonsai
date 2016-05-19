import json
from pprint import pprint

data = {}
data['task'] = 'release'
data['user_id'] = 1
data['galaxy_id'] = 1
data['angle'] = 45.5

json_data = json.dumps(data)

pprint(json_data)

