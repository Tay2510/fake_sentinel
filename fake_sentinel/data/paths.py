from pathlib import Path

PREPROCESSED_FILE_DIR = Path(__file__).parent / 'resources'

DFDC_DATAFRAME_FILE = PREPROCESSED_FILE_DIR / 'dfdc.csv'

DFDC_TRAIN_VIDEO_DIR = '/home/jeremy/data/kaggle/dfdc/train/'

FACENET_RESULT_DIR = '/home/jeremy/data/kaggle/dfdc_faces_100_frames'

FACE_CROP_DIR = '/home/jeremy/data/kaggle/dfdc_face_crops'

FACE_CROP_TABLE = PREPROCESSED_FILE_DIR / 'crop_table.pkl'
