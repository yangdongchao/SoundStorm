import json
import os

# 获取当前脚本的所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# JSON 文件的文件名
json_filename = "ontology.json"

# 构建 JSON 文件的完整路径
json_path = os.path.join(script_dir, json_filename)

id_name_dict = {}

with open(json_path, 'r') as f:
    json_items = json.load(f)
# '/m/0dgw9r' -> 'Human sounds' and etc.
for item in json_items:
    id_name_dict[item['id']] = item['name']
