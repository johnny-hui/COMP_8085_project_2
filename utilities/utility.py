import getopt
import sys
import pandas as pd


def test_data_to_json(X_test, y_test, file_path: str):
    """
    Coverts the test data into a json file.

    @param X_test:
        The independent variable

    @param y_test:
        The dependent target variable(s)

    @param file_path:
        The path of the JSON file to be saved

    @return: None
    """
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_json(file_path.replace('.json', '_test.json'), orient='records', lines=True)


def parse_arguments():
    """
    Parse the command line for arguments.

    @return: classifier, experiment, json_file_path, inference
        Arguments for the initialization of program
    """
    # Initialization
    technique, experiment, json_file_path, mode = "", "", "", ""

    # GetOpt Arguments
    arguments = sys.argv[1:]
    opts, user_list_args = getopt.getopt(arguments, 't:e:f:m:')

    # If no arguments
    if len(opts) == 0:
        sys.exit("[+] ERROR: No arguments were provided!")

    for opt, argument in opts:
        if opt == '-t':  # For modelling technique
            if argument == "neural_network":
                technique = argument
            elif argument == "probabilistic":
                technique = argument
            else:
                sys.exit("[+] ERROR: Invalid modelling technique provided! (-t option)")

        if opt == '-e':  # For experiment number
            try:
                experiment = int(argument)
                if not (1 <= experiment <= 3):
                    sys.exit("[+] ERROR: Experiment must be an integer from 1 - 3 (-e option)")
            except ValueError:
                sys.exit("[+] ERROR: Invalid data type provided for experiment number (-e option)!")

        if opt == '-f':  # For json file path
            try:
                with open(argument, "r"):
                    json_file_path = argument
            except FileNotFoundError:
                sys.exit("[+] ERROR: File (-f option) does not exist in the given path! ({})".format(argument))

        if opt == '-m':  # For mode type (training or inference)
            if argument == "training":
                mode = argument
            elif argument == "inference":
                mode = argument
            else:
                sys.exit("[+] ERROR: Invalid mode provided! (-m option)")

    # Check if parameters are filled out
    if len(technique) == 0:
        sys.exit("[+] ERROR: A modelling technique not specified! (-t option)")

    if len(str(experiment)) == 0:
        sys.exit("[+] ERROR: Experiment number not specified! (-e option)")

    if len(json_file_path) == 0:
        sys.exit("[+] ERROR: A JSON file path not specified! (-f option)")

    if len(mode) == 0:
        sys.exit("[+] ERROR: The mode (training or inference) was not specified! (-m option)")

    return technique, experiment, json_file_path, mode
