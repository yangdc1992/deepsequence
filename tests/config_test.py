import json
from context import deepsequence
from deepsequence.utils import conf_parse
from deepsequence.config import Params

# old_params = conf_parse('/mnt/d/Project/deepsequence/examples/parameters.ini')
# print(old_params)
# with open('/mnt/d/Project/deepsequence/examples/parameters.json', 'w') as file:
#     json.dump(old_params, file, indent=4)

params = Params('/mnt/d/Project/deepsequence/examples/parameters.json')
print(params.dict)

params.test = 'test'
print(params.dict)