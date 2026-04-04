python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python3 -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python3 data/cached_challenge_fineweb.py --variant sp1024
git config --global user.email "fangyu67@gmail.com"
git config --global user.name "Yu Fang"