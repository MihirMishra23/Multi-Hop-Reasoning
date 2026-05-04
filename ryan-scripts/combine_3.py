import json
sft_data = {}
file_path1=""
file_path2=""
db_path=""
for path in [file_path1, file_path2]:
    with open(path) as f:
        data = json.loads