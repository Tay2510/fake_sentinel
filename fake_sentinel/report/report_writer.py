import nbformat as nbf
from pathlib import Path

from fake_sentinel.report.converter.html_converter import convert_notebook_to_html
from fake_sentinel.report.notebook import *


def write_notebook_report(result_dir_path, report_data, filename='report.ipynb'):

    nb = nbf.v4.new_notebook()

    nb['cells'] = [
        nbf.v4.new_code_cell(SET_UP_CODE.format(Path(result_dir_path).absolute())),
        nbf.v4.new_markdown_cell(SECTION_TITLE.format(title=Path(result_dir_path).name)),

        nbf.v4.new_markdown_cell(SECTION_CONFIGS_HEADER),
        nbf.v4.new_code_cell(SECTION_CONFIGS_CODE),

        nbf.v4.new_markdown_cell(SECTION_EVALUATION_HEADER),
        nbf.v4.new_code_cell(SECTION_EVALUATION_CODE.format(eval_loss=report_data['eval_loss'],
                                                            eval_time=report_data['eval_time'])),
        nbf.v4.new_markdown_cell(SECTION_TRAINING_HEADER),
        nbf.v4.new_code_cell(SECTION_TRAINING_CODE),

    ]

    nbf.write(nb, str(filename))

    convert_notebook_to_html(str(filename))
