
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.model import model, DEVICE
from app.preprocessing import preprocess_image, CLASS_NAMES


app = FastAPI(
    title="Alzheimer's MRI Classification API",
    description=(
        "Research prototype for classifying brain MRI images. "
        "This API is not intended for clinical diagnosis."
    ),
    version="1.0.0"
)
# Frontend configuration
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


@app.get("/ui", include_in_schema=False)
def frontend():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/")
def home():
    return {
        "message": "Alzheimer's MRI Classification API is running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "DenseNet121",
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    
    # Read and validate uploaded image
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image file is too large. Maximum size is 10 MB."
        )
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()

        # Reopen after verification
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please upload a valid image file."
        )


    image_tensor = preprocess_image(image).to(DEVICE)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_class = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, predicted_class].item())

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "class_id": predicted_class,
        "confidence": round(confidence, 4),
        "message": (
            "Research output only. This result is not a medical diagnosis."
        )
    }
