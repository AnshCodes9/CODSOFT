# Movie Rating Prediction with Python

## Overview
This project aims to predict the rating of movies based on various features such as genre, director, actors, and more. The goal is to develop a machine learning model using regression techniques that can accurately estimate movie ratings based on historical data.

By analyzing the factors influencing movie ratings, we explore data analysis, preprocessing, feature engineering, and machine learning techniques to provide valuable insights and predictions for movie ratings given by users or critics.

## Project Structure

- **`data/`**: Contains the movie dataset with features such as `Name`, `Year`, `Duration`, `Genre`, `Rating`, `Votes`, `Director`, `Actor 1`, `Actor 2`, `Actor 3`.
- **`notebooks/`**: Jupyter notebooks for data exploration, feature engineering, and model building.
- **`scripts/`**: Python scripts for training and evaluating models.
- **`models/`**: Pre-trained models and results.
- **`README.md`**: This file.

## Dataset
The dataset contains the following columns:
- **`Name`**: The title of the movie.
- **`Year`**: The release year of the movie.
- **`Duration`**: The duration of the movie in minutes.
- **`Genre`**: The genre(s) of the movie.
- **`Rating`**: The rating score given to the movie by users or critics (target variable).
- **`Votes`**: The number of votes the movie received.
- **`Director`**: The director of the movie.
- **`Actor 1`, `Actor 2`, `Actor 3`**: The top-billed actors in the movie.

## Key Features of the Project

### 1. **Data Preprocessing**
   - Handling missing values.
   - Encoding categorical variables like genre, director, and actors.
   - Feature scaling (if necessary).
   
### 2. **Feature Engineering**
   - Converting genres, director names, and actors into numerical features using techniques such as one-hot encoding or label encoding.
   - Creating new features, such as the average rating of the director's past movies or the genre's popularity.

### 3. **Modeling**
   - Regression techniques such as:
     - Linear Regression
     - Ridge and Lasso Regression
     - Decision Trees
     - Random Forest
     - Gradient Boosting (XGBoost, LightGBM)
   - Evaluating models using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.

### 4. **Evaluation**
   - Comparing model performances.
   - Cross-validation to ensure the model's robustness.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/movie-rating-prediction.git
   cd movie-rating-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate   # For Windows: env\Scripts\activate
   ```

3. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Exploration and Preprocessing:**
   Open the Jupyter notebooks in the `notebooks/` directory to explore the data, clean it, and perform feature engineering.
   ```bash
   jupyter notebook
   ```

2. **Model Training and Evaluation:**
   Use the Python scripts in the `scripts/` directory to train and evaluate the models. For example:
   ```bash
   python scripts/train_model.py
   ```

3. **Prediction:**
   Once the model is trained, you can use it to predict the ratings of new movies:
   ```python
   from models.predict import predict_rating
   predict_rating(movie_features)
   ```

## Results
The final model achieved an RMSE of `X.XX` and an R-squared score of `X.XX` on the test set, indicating its ability to predict movie ratings with reasonable accuracy. The feature importance analysis revealed that factors such as `Genre`, `Director`, and `Votes` play a significant role in determining movie ratings.

## Future Improvements
- Incorporate more complex NLP techniques for analyzing text-based features like movie descriptions or reviews.
- Use deep learning models like neural networks to improve predictions.
- Explore more feature interactions to enhance the model's performance.

