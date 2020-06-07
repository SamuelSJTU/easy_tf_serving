import os
import sys
import time

from easy_tf_serving import serving

if __name__ == '__main__':
    # main server process
    serving_process = serving.TFServingService()
    serving_process.start()

    while True:
        print('main thread running')
        time.sleep(100)

