from setuptools import find_packages,setup
from typing import List

hypen="-e ."
def get_requirements(file_path:str)->List[str]:   #convert the requirements.txt text file to a list
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()   #read the entire file_path (here: requirements.txt)
        requirements=[req.strip() for req in requirements]   #remove spaces

        if hypen in requirements:
            requirements.remove(hypen)

    return requirements

setup(
    name='PROJECT2',
    version='0.0.1',
    author='Amar',
    author_email='amarssrh@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)