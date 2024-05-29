# functional_demo.py #
import tkinter as tk
from interactive_demo import InteractiveDemo
from cauldron_app import CauldronApp
from logging_util import logger
from util import db_path, llm_model

def demo():
    logger.info(">> Cauldron Functional Demo <<")
    root = tk.Tk()
    CauldronDemo = InteractiveDemo(root, CauldronApp(db_path, llm_model, verbose=True))
    logger.info("Cauldron Functional Demo Initialized")
    root.protocol("WM_DELETE_WINDOW", CauldronDemo.on_close)  # Bind the close event
    root.mainloop()

if __name__ == "__main__":
    demo()