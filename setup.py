from setuptools import setup, find_packages

setup(
    name='caldron',
    version='0.1.0',
    packages=find_packages(),
    license='MIT',
    long_description=open('README.md').read(),
    python_requires='>=3.10',
    install_requires=[
        'langchain>=0.1.20',
        'langchain-openai>=0.1.6',
        'langchain-community>=0.0.38',
        'langgraph>=0.0.49',
        'openai>=1.28.1',
        'networkx',
        'matplotlib',
        'pydantic>=1.10',
        'python-dotenv',
        'sqlalchemy',
        'ujson',
        'recipe-scrapers',
        'tavily-python',
        'requests',
        'beautifulsoup4',
    ]
)