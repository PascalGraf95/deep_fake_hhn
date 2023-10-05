import subprocess
import shlex

command = "ffmpeg -y -i target.mp4 -ss 0.0 -t 11.75 -filter:v crop=210:210:118:51, scale=256:256 crop.mp4"
subprocess.run(shlex.split(command))