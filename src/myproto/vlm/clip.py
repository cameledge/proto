from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def build_clip(model_name="openai/clip-vit-base-patch32", device="cpu"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def classify_image(image_path, labels, model, processor, device="cpu"):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return {label: float(prob) for label, prob in zip(labels, probs[0])}

if __name__ == "__main__":
    model, processor = build_clip()
    labels = ["a dog", "a cat", "a car"]
    results = classify_image("example.jpg", labels, model, processor)
    print(results)
