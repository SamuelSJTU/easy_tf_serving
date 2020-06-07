## easy_tf_serving
tf serving deploy exmaple

## related project
- https://github.com/tensorflow/serving
- https://github.com/shenweichen/DeepCTR

## tutorial
1. prepare tf serving environment

install tf serving
```
sh tf_serving_install.sh  # or use official tutorial
```
2. start tf serving
```
sh run.sh
```
automatic load model for serving service, start restful and grpc server

3. use client 
```
python client/client_grpc.py
python client/client_restful.py
```