FOLDERS=$(find ../../data/runs -mindepth 1 -maxdepth 1 -type d -exec du -ms {} + | awk '$1 <= 10' | cut -f 2-)
rm -r $FOLDERS

FOLDERS=$(find ../../data/runs/MNIST -mindepth 1 -maxdepth 1 -type d -exec du -ms {} + | awk '$1 <= 6' | cut -f 2-)
rm -r $FOLDERS
