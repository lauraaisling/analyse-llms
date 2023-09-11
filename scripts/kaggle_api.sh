#!/bin/bash

mkdir -p ~/.kaggle
touch ~/.kaggle/kaggle.json

echo -n "Your kaggle username: "
read -r username

echo -n "Your api key: "
read -r api_key

PYCMD=$(cat <<EOF
import json

api_token = {"username":"$username","key":"$api_key"}
with open('/home/laura/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

EOF
)

python -c "$PYCMD"

chmod 600 ~/.kaggle/kaggle.json 