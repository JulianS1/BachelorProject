# Bayes_test (Baschelor Project)

# ***Read***

The data is never pushed to GitHub to preserve privacy. In order for the project to run you must add some folders.


*You therefore must create the data -> (preprocessed, rawData) folders yourself and insert the raw harbour dataset(s) in the rawData folder.

# Getting Started
To get started with this project:

Use some IDE such as 'VisualStudio Code' or 'Pycharm'

### Environment Setup
1. **Clone this project to your computer**:
    ```bash
    git clone https://github.com/JulianS1/BachelorProject.git
    ```

2. **Use Conda Environment**: To ensure consistent dependencies, it’s recommended to create the environment using the provided `bayes_environment.yml` file. Run the following command in your terminal:
    ```bash
    conda env create -f bayes_environment.yml
    ```
    This will create a Conda environment with all the required packages, matching the environment in which the project was developed.

3. **Activate the Environment**:
    Follow the terminal prompts, or enter the following into your terminal:

    ```bash
    conda activate pymc-bart
    ```


4. **Run the Project**:

    - Navigate to the directory containing main.py
    '''bash
    cd src/data
    '''

    - Run the main.py file
    '''bash
    python main.py
    '''
