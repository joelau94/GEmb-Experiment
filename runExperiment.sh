#!/bin/sh

path=$1

rm -rf models/*
mkdir -p models
./scripts/train_pos_en.sh $path
