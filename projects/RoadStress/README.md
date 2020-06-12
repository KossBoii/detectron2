# Clone the github file
git clone https://github.com/KossBoii/detectron2.git

# Create new Anaconda Environments to run the code:
conda create --name py3 python=3.7.0\
conda activate py3\
pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html \
pip install cython pyyaml==5.1\
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
pip install opencv-python\
pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html \
pip install scikit-image

conda install -c anaconda svn\
svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_new 

# Change to the main working directory
cd detectron2/projects/RoadStress/

# Run the training script
- FOR HPC:\
sbatch running_script/model_plain_train.sh

- FOR USUAL LAPTOP:\
conda activate py3\
python3 plain_train_net.py train --weights=coco --dataset=./roadstress_new/

# Run the evaluation script
<u>Note:</u> First go into the output file and get the folder name: Ex:06102020121824/model_final.pth
- FOR HPC:\
sbatch running_script/model_plain_eval.sh

- FOR USUAL LAPTOP:\
conda activate py3\
python3 predict.py --dataset="./roadstress_new/" --weights="./output/06102020121824/model_final.pth" --threshold=0.7
