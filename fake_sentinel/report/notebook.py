SET_UP_CODE = """
%%capture
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from fake_sentinel.report.notebook_kernel import *

result_dir_path = '{}'
"""

SECTION_TITLE = "# {title}"

SECTION_CONFIGS_HEADER = "## Configurations"
SECTION_CONFIGS_CODE = "display_configs()"

SECTION_TRAINING_HEADER = "## Training History"
SECTION_TRAINING_CODE = "display_training_curve(result_dir_path)"

SECTION_EVALUATION_HEADER = "## Evaluation Result"
SECTION_EVALUATION_CODE = "display_evaluation({eval_loss}, {eval_time})"

SECTION_MODEL_HEADER = "## Model"
SECTION_MODEL_CODE = "display_model()"
