from setuptools import setup, find_packages

setup(
    name='ebakery',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'numpy',
        'pandas',
        'networkx',
        'matplotlib',
        'scikit-learn',
        'spacy',
        'nltk',
        'gensim',
        'requests',
        'sqlalchemy',
        'python-dotenv',
    ],
    entry_points={
        'console_scripts': [
            'bipartite=ebakery.scripts.bipartite:main',
            'class_defs=ebakery.scripts.class_defs:main',
            'food_data=ebakery.scripts.food_data:main',
            'genai_tools=ebakery.scripts.genai_tools:main',
            'hmm_test=ebakery.scripts.hmm_test:main',
            'recipe_collector=ebakery.scripts.recipe_collector:main',
            'recipe_state_clusters=ebakery.scripts.recipe_state_clusters:main'
        ]
    }
)