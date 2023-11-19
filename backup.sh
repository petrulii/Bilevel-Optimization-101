#!/usr/bin/env bash
USER=ipetruli
SERVER=edgar
FOLDER=/home/clear/ipetruli/projects/bilevel-optimization/src

rsync -Pavu --exclude={'wandb', '*.stdout', '*.stderr'} $USER@$SERVER:$FOLDER .
git add -u
git commit -m "daily updates"
git push
