echo "Testing..."
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm -r ./tiny-imagenet-200/test
python3 tiny_imagenet/val_format.py
find . -name "*.txt" -delete
cp -r tiny-imagenet-200 data/dataset/