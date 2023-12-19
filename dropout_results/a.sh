mkdir -p 1
mkdir -p 2
mkdir -p 3
mkdir -p 4
echo -n "brisingr@" | xclip -sel c
rsync vivian@10.250.1.58:py/ 1/ -a --exclude=resnet1.pth
rsync vivian@10.250.1.58:py2/ 2/ -a --exclude=resnet1.pth
rsync vivian@10.250.1.58:py3/ 3/ -a --exclude=resnet1.pth
rsync vivian@10.250.1.58:py4/ 4/ -a --exclude=resnet1.pth
python3 make_graph.py
