cd /code
PATH=/opt/conda/bin/:$PATH

python --version
make
python setup.py bdist_wheel

