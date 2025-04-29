# I used Python 3.12.3 and the following commands to install what's needed
# The requirements.txt file was generated afterwards

# Update pip
python3 -m pip install -U pip

# Torch and torchvision 
python3 -m pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Matplotlib
python3 -m pip install -U matplotlib
