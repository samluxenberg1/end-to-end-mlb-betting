import yaml

def load_config(file_path):

    try: 
        with open('config.yml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("YAML file not found")
    except yaml.YAMLError as e: 
        print(f"Error parsing YAML: {e}")