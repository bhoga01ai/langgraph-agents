import importlib.util
import sys
import os

# Import the get_env_keys function from 4_langgraph_tools_agent.py
spec = importlib.util.spec_from_file_location("langgraph_tools_agent", "4_langgraph_tools_agent.py")
module = importlib.util.module_from_spec(spec)
sys.modules["langgraph_tools_agent"] = module
spec.loader.exec_module(module)

# Get the get_env_keys function
get_env_keys = module.get_env_keys

# Test the get_env_keys tool
print("\nTesting get_env_keys tool:\n")
result = get_env_keys.run({})
print(result)