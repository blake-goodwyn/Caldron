from config import validate_required_keys
from cauldron_app import CaldronApp
from util import db_path, llm_model

validate_required_keys()
app = CaldronApp(db_path, llm_model)