from typing import List

HYPEN_E_DOT = "-e ."
def get_requirement(file_path:str)-> List[str]:
    '''
    This function return a list of requirements
    '''
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements ]
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements

if __name__ == "__main__":
    print(get_requirement('requirement.txt'))