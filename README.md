<pre><code>conda create -n RL python=3.10 -y
echo 'alias rl="conda activate RL"' >> ~/.bashrc
source ~/.bashrc
rl

sudo apt-get update
sudo apt-get install -y ffmpeg
  
pip install numpy pandas matplotlib pathlib pillow gymnasium stable-baselines3 imageio tensorboard ffmpeg opencv-python
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121</code></pre>
