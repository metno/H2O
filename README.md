# H2O
H2O source code

Enter H2O source code directory

'''
source /modules/centos7/conda/Feb2021/etc/profile.d/conda.sh
conda activate production-04-2021

python3 -m pip install poetry --user
poetry env use /modules/centos7/conda/Feb2021/envs/production-04-2021/bin/python
poetry install
poetry run pip install --force-reinstall shapely --no-binary shapely

poetry shell

./Pyflwdir/example.py
'''