from models.MaxEntClassifier import MaxEntClassifier
from models.MultinomialNB import MultinomialNB
from utilities.utility import parse_arguments

if __name__ == '__main__':
    name, experiment, json_file_path, mode = parse_arguments()

    if name == "johnny" and experiment == 1 and mode == "training":
        MultinomialNB.perform_experiment_one_training(file_path=json_file_path)

    if name == "johnny" and experiment == 1 and mode == "inference":
        MultinomialNB.perform_experiment_one_inference(file_path=json_file_path)

    if name == "johnny" and experiment == 2 and mode == "training":
        MultinomialNB.perform_experiment_two_training(file_path=json_file_path)

    if name == "johnny" and experiment == 2 and mode == "inference":
        MultinomialNB.perform_experiment_two_inference(file_path=json_file_path)

    if name == "johnny" and experiment == 3 and mode == "training":
        MaxEntClassifier.perform_experiment_three_training(file_path=json_file_path)

    if name == "johnny" and experiment == 3 and mode == "inference":
        MaxEntClassifier.perform_experiment_three_inference(file_path=json_file_path)

    if name == "johnny" and experiment == "other" and mode == "training":
        MaxEntClassifier.perform_base_experiment_training(file_path=json_file_path)

    if name == "johnny" and experiment == "other" and mode == "inference":
        MaxEntClassifier.perform_base_experiment_inference(file_path=json_file_path)
