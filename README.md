This project aims to reproduce the results from the classic paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

You need to install the [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
To install using pip, run
```shell
pip install ale-py
```


In the same directory, create a folder called "roms" that contains the Atari 2600 roms for the games. You can obtaining the roms by downloading from http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html and extracting the .rar file.
To import the roms to the Acade Learning Environment, run
```shell
ale-import-roms roms/
```
    
After training the model using atari-DQN.py, you can render a video from the screenshots using e.g. ffmpeg with
```shell
game="pong"
episode="1"
ffmpeg -r 1 -pattern_type glob -r 25 -i "${game}_screenshots/${game}_${episode}_screenshot_*.png" -c:v libx264 "${game}_${episode}.mp4"
```
