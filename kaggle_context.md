### Information about this Kaggle competition

1. Official Competition Description
Here is the description of the competition, sourced directly from the provided Kaggle link.
Competition: Predict Calorie Expenditure (Playground Series - Season 5, Episode 5)
The goal of this competition is to predict the number of calories burned during a workout. This is a regression problem where you will predict a continuous target value.
Dataset Details
The dataset for this competition was generated from a deep learning model trained on the "Calories Burnt Prediction" dataset. The feature distributions are similar, but not identical, to the original data.
Files:
train.csv: The training dataset. The Calories column is the target variable.
test.csv: The test dataset. Your task is to predict the Calories for each entry.
sample_submission.csv: A sample submission file showing the correct format.
Data Fields:
id: A unique identifier for each entry.
Gender: Gender of the participant.
Age: Age of the participant in years.
Height: Height of the participant in cm.
Weight: Weight of the participant in kg.
Duration: Workout duration in minutes.
Heart_Rate: Average heart rate in beats per minute during the workout.
Body_Temp: Body temperature in Celsius during the workout.
Calories: Total calories burned (the target variable).
Evaluation Metric
Submissions are evaluated on the Root Mean Squared Logarithmic Error (RMSLE). This metric is useful for regression tasks where the target has a wide range of values and penalizes underprediction more heavily than overprediction.



For data, look into data subfolder. Never go outside of this repo! Use subsample files except for generating final predictions for submission.

