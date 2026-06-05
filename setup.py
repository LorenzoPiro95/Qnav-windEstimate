from setuptools import setup, find_packages


setup(
    name="qnav",
    version="1.1.0",
    description="qnav: Qlearning for Olfactory Navigation",
    python_requires='>=3.9',
    setup_requires=[
        'setuptools>=18.0'
    ],
    packages=find_packages(),
    install_requires=['numpy','tqdm','matplotlib'],
    include_package_data=True,
)