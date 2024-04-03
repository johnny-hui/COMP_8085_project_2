from models.MultinomialNB import MultinomialNB
from utilities.utility import parse_arguments

if __name__ == '__main__':
    technique, experiment, json_file_path, mode = parse_arguments()

    if mode == 'training' and technique == "probabilistic" and experiment == 1:
        MultinomialNB.perform_experiment_one_training(file_path=json_file_path)

    if mode == 'inference' and technique == "probabilistic" and experiment == 1:
        print("Loading pickled model...")

