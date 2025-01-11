<pre><code>conda create -n RL python=3.10 -y
echo 'alias rl="conda activate RL"' >> ~/.bashrc
source ~/.bashrc
rl

pip install gym stable_baselines3 env pathlib pandas matplotlib torch</code></pre>
