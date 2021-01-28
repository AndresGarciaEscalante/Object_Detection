import os

# Install the Object Detection 
os.system('pip install -r Object_Detection_Urban_Environment/build/requirements.txt')
os.system('pip install tensorflow-gpu')
os.system('git clone https://github.com/tensorflow/models.git')
os.system('conda install -c anaconda protobuf')
os.chdir("models/research")
os.system('protoc object_detection/protos/*.proto --python_out=.')
os.system('pip install cython')
os.system('cp object_detection/packages/tf2/setup.py .')
os.system('pip install --upgrade pip==20.2.2')
os.system('python -m pip install --use-feature=2020-resolver .')
os.system('pip install git+https://github.com/google-research/tf-slim')
os.system('pip install tf-models-official')
os.system('pip install matplotlib==3.1.0')
os.system('python object_detection/builders/model_builder_tf2_test.py')

# Install the Google Cloud and syncronize the user's account
os.chdir("../..")
os.system('echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list')
os.system('curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -')
os.system('sudo apt-get update && sudo apt-get install google-cloud-sdk')
os.system('gcloud auth login')
os.system('gsutil ls gs://waymo_open_dataset_v_1_2_0_individual_files/')

# Install other dependencies 
os.system('pip3 install --upgrade tensorflow==2.3.1')
os.system('mkdir -p data/training data/validation data/testing backups')