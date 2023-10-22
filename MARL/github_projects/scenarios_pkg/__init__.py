import os
import os.path as osp
from importlib.machinery import SourceFileLoader

def load(directory_path):
    loaded_modules = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.py') and filename != '__init__.py':
            full_path = osp.join(directory_path, filename)
            module_name = osp.splitext(filename)[0]  # Get module name without the '.py' extension
            loaded_module = SourceFileLoader(module_name, full_path).load_module()
            loaded_modules[module_name] = loaded_module
    return loaded_modules

# Direct path to scenarios_pkg directory
directory_path = r"/home/yungisimon/MARL/MARL/github_projects/scenarios_pkg"

# Load all Python files in the directory
loaded_modules = load(directory_path)
