#!/bin/bash

export DATASET_PATH=../training.csv

python predict.py

WEEKLY_EVALUATION=true python predict.py
