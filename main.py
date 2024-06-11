import os
from train import train

# create directories for results
def create_result_directories(result_path):
    subdirectories = ['logs', 'plots', 'setup']

    try:
        # Crea la cartella 'result'
        os.makedirs(result_path, exist_ok=True)
        print(f"Directory 'result' creata in {result_path}")
        
        # Crea le sotto-cartelle
        for subdirectory in subdirectories:
            subdirectory_path = os.path.join(result_path, subdirectory)
            os.makedirs(subdirectory_path, exist_ok=True)
            print(f"Sotto-cartella '{subdirectory}' creata in {subdirectory_path}")
    
    except Exception as e:
        print(f"Errore nella creazione delle directory: {e}")



if __name__ == "__main__":
    result_path = './result'

    if not os.path.exists(result_path):
        create_result_directories(result_path)

    train()