

import os
import sys
import json
import time
import grpc
import subprocess
import setproctitle

from multiprocessing import Pipe
from multiprocessing import Process
from multiprocessing import Manager

from tensorflow_serving.apis import model_service_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.config import model_server_config_pb2

class TFServingService(Process):
    def __init__(self):
        super().__init__()
        return

    def __initialize(self):
        setproctitle.setproctitle('tf-serving')
        tf_serving_config_file = './config/tf_serving.json'
        with open(tf_serving_config_file, 'r') as fp:
            serving_config = json.load(fp)
        grpc_port = serving_config['gprc_port']  
        use_batch = serving_config['use_batch'] 

        options = [('grpc.max_send_message_length', 1024 * 1024 * 1024), ('grpc.max_receive_message_length', 1024 * 1024 * 1024)]
        
 
        model_config_file = './config/tf_serving_model.conf'
        cmd = 'tensorflow_model_server --port={} --rest_api_port={} --model_config_file={}'.format(grpc_port-1, grpc_port, model_config_file)
        if use_batch:
            batch_parameter_file = './config/batching.conf'
            cmd = cmd + ' --enable_batching=true --batching_parameters_file={}'.format(batch_parameter_file)
        self.serving_cmds = []
        self.serving_cmds.append(cmd)
        print(self.serving_cmds)
        system_env = os.environ.copy()
        self.process = subprocess.Popen(self.serving_cmds, env=system_env, shell=True)

        ## start grpc 
        self.grpc_channel = grpc.insecure_channel('127.0.0.1:{}'.format(grpc_port-1), options=options)
        self.grpc_stub = model_service_pb2_grpc.ModelServiceStub(self.grpc_channel)

        # wait for start tf serving
        time.sleep(3)

        # reload model
        model_server_config = model_server_config_pb2.ModelServerConfig() 
        config_list = model_server_config_pb2.ModelConfigList()

        model_config_json_file = './config/model.json'
        with open(model_config_json_file, 'r') as fp:
            model_configs = json.load(fp)
        model_dir = model_configs['model_dir']
        model_list = model_configs['models']
        for model_info in model_list:
            print(model_info)
            model_config = model_server_config_pb2.ModelConfig()
            model_config.name = model_info['name']
            model_config.base_path = os.path.abspath(os.path.join(model_dir, model_info['path']))
            model_config.model_platform = 'tensorflow'
            config_list.config.append(model_config)

        model_server_config.model_config_list.CopyFrom(config_list)
        request = model_management_pb2.ReloadConfigRequest()
        request.config.CopyFrom(model_server_config)

        grpc_response = self.grpc_stub.HandleReloadConfigRequest(request, 30)
        print(grpc_port)
        return 

    def run(self):
        self.__initialize()
        while True:
            time.sleep(1)
        return