eval "$(/opt/conda/bin/conda shell.bash hook)"
source activate py37cu102
python /code/travis/create_build_script.py
bash /code/travis/do_build.sh