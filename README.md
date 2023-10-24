<h1>Fraud Detection System for SafeGuard Bank</h1>
<img src="data/images/SAFEGUARD.png" alt="SafeGuard Bank Logo" width="530" />
<p>Welcome to SafeGuard Bank, a leading financial institution committed to ensuring the security and integrity of your transactions. Our cutting-edge fraud detection system employs advanced machine learning algorithms and state-of-the-art data preprocessing techniques to protect your assets and provide you with peace of mind.</p>
<p>In this project, we have developed a robust fraud detection system specifically designed for SafeGuard Bank. By harnessing the power of various machine learning algorithms and leveraging sophisticated data preprocessing techniques, we have created a highly accurate and efficient system that detects fraudulent activities.</p>
<p>Our team of data scientists and engineers has meticulously fine-tuned the system by performing hyperparameter tuning, enabling it to deliver precise and reliable results. Through extensive experimentation and analysis using Jupyter notebooks, we have identified the random forest algorithm as the optimal choice for fraud detection at SafeGuard Bank.</p>

<p>
    <strong>Project Organization</strong>
    <br>
    To ensure seamless development and ease of use, our project is structured into the following directories:
    <br>
    <br>
    <strong>configuration:</strong> Contains configuration files for the project.
    <br>
    <strong>data:</strong> Contains the dataset used for training and testing the fraud detection system.
    <br>
    <strong>log:</strong> Stores log files generated during the execution of the system.
    <br>
    <strong>notebooks:</strong> Includes Jupyter notebooks used for experimentation and development.
    <br>
    <strong>model:</strong> Holds the trained model and related files.
    <br>
    <strong>scripts:</strong> Contains scripts for running different parts of the project.
    <br>
    <strong>tests:</strong> Holds the mockup datasets, fixtures and tests for each class.
    <br>
</p>

## Project Structure

```
├── configuration
│   ├── config.yaml
│   └── hyperparameters.yaml
├── data
│   ├── etl
│   │   ├── test.csv
│   │   └── train.csv
│   ├── images
│   │   └── SAFEGUARD.png
│   ├── predictions
│   │   └── predictions.csv
│   ├── preprocessed
│   │   ├── X_test.csv
│   │   ├── X_train.csv
│   │   ├── y_test.csv
│   │   └── y_train.csv
│   └── raw
│       └── fraud_dataset.csv
├── Makefile
├── LICENSE
├── log
│   ├── ETL.log
│   ├── FeatureEngineering.log
│   ├── ModelTraining.log
│   └── Predictor.log
├── model
│   ├── model_random_forest_1.0.pkl
│   └── model_random_forest_2.0.pkl
├── my_etl_app.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
├── notebooks
│   └── fraud_detection.ipynb
├── README.md
├── requirements.txt
├── scripts
│   ├── __init__.py
│   ├── etl.py
│   ├── FeatureEngineering.py
│   ├── ModelTraining.py
│   └── Predictor.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── empty_file.csv
    ├── missing_columns.csv
    └── test_ETL.py
    └── test_feature_engineering.py
    └── test_model_training.py
    └── test_model_predictor.py
    └── test.parquet
```

<p>
    <strong>Step-by-Step Guide:</strong>
    <br>
    Follow these steps to test your project and use the provided Makefile:
    <br>
    <ol>
        <li><strong>Initialize the Environment:</strong>
            <br>
            To set up your environment, run the following command:
            <pre>make init</pre>
            This will install the required dependencies for your project.
        </li>
        <li><strong>Run Tests:</strong>
            <br>
            You can run tests using the following command:
            <pre>make test</pre>
            This command will execute your test suite using pytest, providing detailed information about test results.
        </li>
    </ol>
</p>

<p>
    <strong>Testing Framework:</strong>
    <br>
    We use pytest, a popular Python testing framework, to perform unit testing on our project. The "tests" folder contains various test files that validate the functionality of different components in our fraud detection system.
</p>


<p>
    <strong>Technologies Leveraged</strong>
    <br>
    At SafeGuard Bank, we harness the power of cutting-edge technologies to deliver the highest level of security and accuracy. Our project incorporates the following technologies:
    <br>
    <br>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vscode/vscode-original.svg" width="60" height="60" vspace="5" hspace="5">
    Visual Studio Code: Our primary integrated development environment (IDE) for seamless coding and project management.
    <br>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original-wordmark.svg" width="60" height="60" vspace="5" hspace="5">
    Kaggle: A platform that enables us to access diverse datasets, collaborate with the data science community, and explore innovative ideas.
    <br>
    <img src="https://pandas.pydata.org/static/img/favicon_white.ico" width="80" height="50" vspace="5" hspace="5">
    Pandas: A powerful data manipulation and analysis library that ensures efficient preprocessing of data for accurate fraud detection.
    <br>
    <img src="https://seeklogo.com/images/S/scikit-learn-logo-8766D07E2E-seeklogo.com.png" width="80" height="60" vspace="5" hspace="5">
    scikit-learn: An essential toolkit for implementing machine learning algorithms, evaluating models, and enhancing the performance of our fraud detection system.
</p>


          
          
