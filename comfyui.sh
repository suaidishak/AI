sudo apt update
sudo apt upgrade -y
sudo apt-get install libgoogle-perftools-dev
sudo snap install ngrok
ngrok authtoken 2YmLJMmvcUf0UzgqM7Mtk9oRhav_88jaJRR3wnBTff9diyZnz
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python3 -m venv venv
source venv/bin/activate
pip install -R requirements.txt
