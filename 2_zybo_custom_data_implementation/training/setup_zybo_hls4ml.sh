#!/bin/bash

cp ./hls4ml_setup/supported_boards.json venv/lib/python3.8/site-packages/hls4ml/templates/supported_boards.json
cp -r ./hls4ml_setup/zybo-z7010 venv/lib/python3.8/site-packages/hls4ml/templates/vivado_accelerator/
