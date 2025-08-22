# wine-quality-classification

This project is a group project from the course ITEC 3040: 
Introduction to Data Analytics. It contains files containing 
the algorithms used for predicting Wine Quality. 

## Installation

Upload files into your preferred IDE. Use package manager pip to install the necessary packages.

```pip install pandas scikit-learn matplotlib seaborn ucimlrepo```

## Usage

Run each python file individually to obtain results. To change the csv file, update the **wine_quality_color** variable
found at the top of the code:

```#select wine quality color data set here ('white' or 'red')
wine_quality_color = 'white' 
# (or wine_quality_color = 'red')

print(f'You have chosen the {wine_quality_color} wine dataset.\n')
```
Note: Don't change the csv file for __naive_bayes_model.py__, as there is a separate csv for it.

General Output:

- K-value Tests (only KNN)
- Classification Report
- Confusion Matrix's (graph)
- Tuple Predictions
- Feature Importance/Mutual Information Score (graph or table)

## Data Source

[UCI Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)

## Credits

[Annaz Mus Sakib](https://github.com/D1Massacre007) - Decision Tree and KNN Models   
Saad Shahid - Naive Bayes Model   
Zoe Linga - Proofreader + Final Touch ups   
[Ali Zafar](https://github.com/alizafarqureshi) - Presentation
