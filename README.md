### Databricks extension for Visual Studio Code and Databricks Asset Bundles (DABs) demo

This repository provides a simple, data science focused example of the Databricks extension for Visual Studio code. It demonstrates how a data scientist can interactively prototype a model training workflow using Visual Studio code with Databricks compute on the backend for Spark operations. The project is then deployed as a Databricks Workflow configured using DABs.

Some considerations...

 - Using the extension, all code involving Spark DataFrames runs on the Databricks cluster. Pure Python code runs locally; this includes things like Pandas DataFrame transformations and training scikit-learn models (see the [docs](https://docs.databricks.com/en/dev-tools/vscode-ext/notebooks.html#run-and-debug-notebook-cells-with-databricks-connect-using-the-databricks-extension-for-visual-studio-code)).
    - Since pure Python code runs locally, relevent packages must be installed locally, for instance, using a conda environment. Optionally, one can replicate relevent ML Runtime packages by referencing the [runtime documentation](https://docs.databricks.com/en/release-notes/runtime/index.html).
    - If larger compute resources are needed to run Python operations than what is available locally, code can be prototyped locally, for instance, using a data subset. The project can then be deployed as a Workflow on Databricks using DABs, where all code (Both Python and Spark) will run on the Databricks cluster (multi-node or single-node).

 - The extension supports running Jupyter notebook cells against an interactive Databricks cluster. Python files (such as a main.py file that imports modules from the project) can also be run.
 - Local training runs can be easily logged to an Experiment in Databricks-hosted MLflow.


To run this demo...
 - Installed and configured [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/index.html)
 - Clone the repository
 - Copy the training data csv file, data/titanic_train.csv, to a UC volume or the DBS. For example...   
    ```
    databricks fs cp titanic_train.csv dbfs:/Shared/temp_data/titanic_train.csv
    ```




Some helpful links: 
Databricks Connect
    - [Documentation on extension and use with Databricks Connect](https://docs.databricks.com/en/dev-tools/vscode-ext/notebooks.html#run-and-debug-notebook-cells-with-databricks-connect-using-the-databricks-extension-for-visual-studio-code)  

DABs  
    - [Configuring databricks asset bundle files](https://docs.databricks.com/en/dev-tools/bundles/settings.html#databricks-asset-bundle-configurations)  
        - See section on [mappings definitions](https://docs.databricks.com/en/dev-tools/bundles/settings.html#mappings)
    - [Variable substitution with DABs](https://docs.databricks.com/en/dev-tools/bundles/variables.html#set-variable-value)

Databricks API (governs resource configuration in DABs)  
    - [Jobs API documentation](https://docs.databricks.com/api/workspace/jobs/create)  
    - [Cluster API documentation](https://docs.databricks.com/api/workspace/clusters/create)