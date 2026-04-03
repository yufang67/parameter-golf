python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024