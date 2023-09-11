#!/bin/bash

mkdir -p ~/.kaggle
touch -p ~/.kaggle/kaggle.json
api_token = {"username":"lauraomahony999","key":"XXXXX"}
with open('/home/laura/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

!chmod 600 ~/.kaggle/kaggle.json