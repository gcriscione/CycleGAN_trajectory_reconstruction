import json

def load_config(config_path='./config/config.json'):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

config = load_config()

if __name__ == "__main__":
    print(config)