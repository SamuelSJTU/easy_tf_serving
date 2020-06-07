#!/bin/bash
ps -aux | grep tensorflow_model_server | awk '{print $2}' | xargs kill -7
ps -aux | grep main.py | awk '{print $2}' | xargs kill -7