import nbformat as nbf

from pathlib import Path
from subprocess import run, PIPE
from nbconvert.preprocessors import ExecutePreprocessor

TEMPLATE_PATAH = Path(__file__).parent / 'clean_source.tpl'


def convert_notebook_to_html(notebook_path):
    execute_notebook(notebook_path)

    command_string = 'jupyter nbconvert --to html {notebook_path} --template {template_path}'

    command = command_string.format(notebook_path=notebook_path,
                                    template_path=TEMPLATE_PATAH.absolute())

    run(command, universal_newlines=True, stdout=PIPE, stderr=PIPE, shell=True)


def execute_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbf.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    ep.preprocess(nb, {'metadata': {'path': ''}})

    with open(notebook_path, 'wt') as f:
        nbf.write(nb, f)
