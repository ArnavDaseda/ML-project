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