from ultralytics import YOLO

# Load model
model = YOLO('models/best.pt')  

# Apply inference
results = model.predict('data/08fd33_4.mp4', save=True) 
print(results[0])  # Display results
print("\n")
for box in results[0].boxes:
    print(box)
  