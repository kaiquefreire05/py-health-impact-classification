# Health Impact Classification

**Author:** Kaíque Freire dos Santos  
**Date:** 2024/06/22

This Python project performs health impact classification using machine learning techniques. The project utilizes libraries such as pandas, numpy, matplotlib, seaborn, scikit-learn, and Keras.

## Introduction

The objective of this project is to build machine learning models that can predict health impacts based on various variables. This type of analysis can be useful for better understanding how different factors influence health and aiding in making informed decisions.

## Project Structure

The project structure is organized as follows:

```
py-health-impact-classification/
├── dataset-variables/ # Directory for dataset
│ └── air-quality-and-health-impact-dataset.zip
│ └── air-quality-health-impact-data.csv
│ └── healthimpact.pkl
├── logs/ # Directory for log files
│ └── info-preprocessing.log
│ └── training-specific.log
├── plots/ # Directory for generated plots
├── creating-model.py # Script for model creation
├── loading-preprocessing-dataset.py # Script for loading and preprocessing data
├── .gitignore # Gitignore configuration file
└── README.md # This README file
```

## Data

The dataset used in this project is located in the `dataset-variables/` directory and contains necessary information for training and evaluating machine learning models. The file `air-quality-and-health-impact-dataset.zip` is extracted to obtain the CSV file `air_quality_health_impact_data.csv`.

## Data Preprocessing

The data undergoes preprocessing, which includes:

1. Loading the data.
2. Handling missing values.
3. Encoding categorical variables.
4. Normalizing numerical data.

The preprocessing script is located in `loading-preprocessing-dataset.py`.

## Model Training

Machine learning models are trained using a dense neural network with the Keras library. The training script is located in `creating-model.py` and includes the use of callbacks such as EarlyStopping and ReduceLROnPlateau.

## Results

Distribution plots, correlation maps, and confusion matrices are generated and saved in the `plots/` directory.

## Contribution

Contributions are welcome! Feel free to open an issue or submit a pull request.

## Licença

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for more details.
