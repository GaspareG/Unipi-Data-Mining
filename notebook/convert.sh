#!/bin/bash

sudo rm CardCardDefault_files/*
sudo jupyter nbconvert --to markdown *.ipynb
