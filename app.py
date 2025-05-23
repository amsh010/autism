import io
import os

import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Import prediction function from AI.py
from AI import predict_new_data, load_model_artifacts, evaluate_model

# Create FastAPI app
app = FastAPI(title="Autism Diagnosis System")

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with file upload form"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Handle file upload and make predictions"""
    try:
        # Read the uploaded file
        contents = await file.read()

        # Process the file based on its extension
        if file.filename.endswith(('.xls', '.xlsx')):
            new_data = pd.read_excel(io.BytesIO(contents))
        elif file.filename.endswith('.csv'):
            new_data = pd.read_csv(io.BytesIO(contents))
        else:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": "Unsupported file format. Please upload an Excel or CSV file."}
            )

        # Make predictions
        predictions, probs = predict_new_data(new_data)

        # Prepare results for display
        results = []
        for idx, (prediction, prob) in enumerate(zip(predictions, probs)):
            # Format probability as percentage
            prob_percent = f"{prob[1] * 100:.2f}%"
            results.append({
                "row": idx + 1,
                "prediction": prediction,
                "probability": prob_percent
            })

        return templates.TemplateResponse(
            "results.html",
            {"request": request, "results": results, "filename": file.filename}
        )

    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": f"Failed to process the file: {str(e)}"}
        )


@app.get("/evaluate", response_class=HTMLResponse)
async def evaluate(request: Request):
    """Evaluate the current model and show results"""
    try:
        metrics = evaluate_model()

        return templates.TemplateResponse(
            "evaluation.html",
            {
                "request": request,
                "metrics": metrics
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": f"Failed to evaluate model: {str(e)}"}
        )


if __name__ == "__main__":
    # Check if model exists
    try:
        model, train_data, test_data, label_encoders = load_model_artifacts()
        print(f"Model loaded. Training data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure the model files exist before running the application.")

    # Run the FastAPI app with Uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
