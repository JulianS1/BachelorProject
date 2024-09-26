# BachelorProject

# ***Read***

The data is never pushed to GitHub to preserve privacy. In order for the project to run you must add some folders.

the structure should be

Parent Directory:

-   data:
    - preprocessed
    - rawData: (Here is where you place the raw dataset)

-   results:

-   src:
    - data: main.py
    utils:
        - models.py
        - Preprocessor.py
        - visualise.py
        - etc
- etc


*You therefore must create the data -> (preprocessed, rawData) folders yourself and insert the raw harbour dataset(s) in the rawData folder.

# Getting Started
To get started with this project:

Use some IDE such as 'VisualStudio Code' or 'Pycharm'

### Environment Setup
1. **Clone this project to your computer**:
    ```bash
    git clone https://github.com/JulianS1/BachelorProject.git
    ```

2. **Use Conda Environment**: To ensure consistent dependencies, it’s recommended to create the environment using the provided `environment.yml` file. Run the following command in your terminal:
    ```bash
    conda env create -f environment.yml
    ```
    This will create a Conda environment with all the required packages, matching the environment in which the project was developed.

3. **Activate the Environment**:
    ```bash
    conda activate BScProject
    ```






    TODO:

ensure not replacing '<' by 0 but by proper values
