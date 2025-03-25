import base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv8 model
model = YOLO("customModel.pt")  # Ensure this model is in the same directory

# Get class names from the model
class_names = model.names  # Dictionary {0: 'person', 1: 'bicycle', 2: 'car', ...}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(file.file)

    # Run YOLOv8 inference
    results = model(image)

    # Convert image to base64 (optional)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Prepare response
    detections = []
    for r in results:
        for i, box in enumerate(r.boxes.xyxy):  # Extract bounding boxes
            class_index = int(r.boxes.cls[i])  # Get class index
            class_name = class_names.get(class_index, "Unknown")  # Get class name
            
            detections.append({
                "bbox": box.tolist(),
                "confidence": float(r.boxes.conf[i]),
                "class_name": class_name
            })

    return JSONResponse({
        "status": "success",
        "detections": detections,
        "image_base64": img_str
    })

# Run the app locally (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
