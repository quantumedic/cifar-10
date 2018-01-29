if[$1 == "reset"]; then rm -rf ./save/*
rm -rf ./logs/*
rm -rf ./dist
python train.py