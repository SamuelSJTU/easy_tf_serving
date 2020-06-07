## apt install
# official install tutorial https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md

pip install tensorflow-serving-api==1.15.0  # match version with tf

echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update && apt-get install tensorflow-model-server

## todo alternative build from source & use docker (bazel needed)