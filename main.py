from fastapi import FastAPI, File, UploadFile # type: ignore
from pydantic import BaseModel # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
import joblib
import io

app = FastAPI()

# Global variables for data and model
df = None
model = None

# Root endpoint
@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Model Training API! Use the endpoints for uploading datasets, preprocessing data, training models, and making predictions."
    }

# Endpoint for uploading dataset
@app.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    global df
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    return {"message": "Dataset uploaded successfully", "columns": list(df.columns)}

# Endpoint for preprocessing data
@app.post("/preprocess-data/")
async def preprocess_data(label_column: str):
    global df
    if df is None:
        return {"error": "Dataset not uploaded"}

    # Drop any unnamed index column
    df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True, errors='ignore')
    
    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill numerical NaNs with mean
    df.fillna("Unknown", inplace=True)  # Fill categorical NaNs with "Unknown"
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=["object"]).columns:
        if column != label_column:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    df[df.select_dtypes(include=["float64", "int64"]).columns] = scaler.fit_transform(df.select_dtypes(include=["float64", "int64"]))
    
    # Save transformations for later use
    joblib.dump(label_encoders, "label_encoders.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    return {"message": "Data preprocessed successfully"}

# Model training request schema
class TrainRequest(BaseModel):
    label_column: str  # Target column (e.g., "MathScore", "ReadingScore", or "WritingScore")

# Endpoint for training model
@app.post("/train-model/")
async def train_model(request: TrainRequest):
    global df, model
    if df is None:
        return {"error": "Dataset not uploaded"}
    
    X = df.drop(columns=[request.label_column])
    y = df[request.label_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_score = mean_squared_error(y_test, y_pred)
    
    # Save model
    joblib.dump(model, "model.pkl")
    
    return {
        "message": f"Model trained successfully for {request.label_column}",
        "mse_score": mse_score,
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist()
    }

# Predict request schema
class PredictRequest(BaseModel):
    data: dict

# Endpoint for making predictions
@app.post("/predict/")
async def predict(request: PredictRequest):
    global model
    if model is None:
        return {"error": "Model not trained"}
    
    try:
        # Load preprocessors and model
        label_encoders = joblib.load("label_encoders.pkl")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        return {"error": "Required model or preprocessor files not found."}

    # Prepare input data
    input_data = pd.DataFrame([request.data])
    for column, le in label_encoders.items():
        if column in input_data.columns:
            input_data[column] = le.transform(input_data[column].astype(str))
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    
    return {"prediction": prediction.tolist()}
