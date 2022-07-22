# The following programs must be installed for this script to work:
# - git
# - docker
# - transmission-cli

DATASET_ARCHIVE="Ch2_002.tar.gz"

# Pulling the NVIDIA DAVE-2 dataset from the Udacity repository.
if [ ! -d "self-driving-car" ] ; then
    git clone https://github.com/udacity/self-driving-car.git
fi

# Pulling the scripts for pre-processing the dataset.
if [ ! -d "udacity-driving-reader" ] ; then
    git clone https://github.com/rwightman/udacity-driving-reader
fi

# Applying changes to pre-processing scripts to use a new Docker image
# as the Udacity image from the original challenge is no longer available.
cd udacity-driving-reader
git apply ../docker_images.diff
cd -

# Pulling down the dataset pre-processing Docker image from the new location.
docker pull jmidwint/udacity-reader

# Torrenting the dataset file (this may take a while).
if [ ! -d "data" ] ; then
   mkdir -p data/compressed_dataset
   mkdir -p data/dataset
fi

# Downloading the dataset. Note that we only use the training dataset
# provided by Udacity as the testing dataset appears to have no labels.
#
# If this Torrent freezes in the seeding phase (i.e. the kill in the post-script doesn't work), feel free to terminate
# and execute the script again as the downloaded file will then be present and the following line shouldn't be executed.
if [ ! -f $DATASET_ARCHIVE ] ; then
	transmission-cli self-driving-car/datasets/CH2/${DATASET_ARCHIVE}.torrent -w data/compressed-dataset -f "kill $(pgrep transmission)"
fi

# Unzipping the dataset.
if [ ! -f "data/compressed-dataset/HMB.txt" ] ; then
	cd data/compressed_dataset
	tar xvzf ${DATASET_ARCHIVE}
	cd -
fi

# Executing the pre-processing script.
if [ ! -f "data/dataset/steering.csv" ]; then
	cd udacity-driving-reader
	chmod +x run-bagdump.sh
	./run-bagdump.sh -i $(readlink -m ../data/compressed-dataset) -o $(readlink -m ../data/dataset)
	cd -
fi

# Creating directory for training checkpoints.
mkdir checkpoints
