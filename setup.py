from setuptools import setup,find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    hyphen_e_dot = "-e ."
    with open(file_path) as file_obj:
        requirements =file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements] 
    if hyphen_e_dot in requirements:
        requirements.remove(hyphen_e_dot)
    return requirements 

setup(
    name="MLproject",
    author="Arnav",
    author_email="arnavdaseda3@gmail.com",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)