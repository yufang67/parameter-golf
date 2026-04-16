python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python3 -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
#cp -r data data2
#.venv/bin/python3 data/cached_challenge_fineweb.py --variant sp1024
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf .venv/bin/python3 data/cached_challenge_fineweb.py --variant sp8192
.venv/bin/pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch2110/
git config --global user.email "fangyu67@gmail.com"
git config --global user.name "Yu Fang"