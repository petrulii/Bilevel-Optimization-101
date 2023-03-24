conda activate bilevel

source gpu_setVisibleDevices.sh
GPUID=0
cd /home/clear/ipetruli/projects/bilevel-optimization/
python3 src/main.py