# src/myproto/pipelines/multimodal.py
from typing import List, Optional, Tuple, Dict
from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image
import torch

class MultiModalPipeline:
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        clip_device: int = -1,
        llm_model_name: str = "gpt2",
        llm_device: int = -1,
        llm_max_length: int = 64,
        llm_do_sample: bool = False,
    ):
        # CLIP
        self.clip_device = "cpu" if clip_device == -1 else f"cuda:{clip_device}"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.clip_device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # LLM text-generation pipeline
        # pipeline accepts device as int: -1 for CPU, 0..N for CUDA device index
        self.llm = pipeline(
            "text-generation",
            model=llm_model_name,
            device=llm_device,
            max_length=llm_max_length,
            do_sample=llm_do_sample,
            truncation=True,
        )

    def image_to_top_labels(self, image_path: str, candidate_labels: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Zero-shot: compute CLIP logits for the candidate labels and return top_k labels with probs.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(text=candidate_labels, images=image, return_tensors="pt", padding=True).to(self.clip_device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # logits_per_image shape: (batch_size=1, num_texts)
            logits = outputs.logits_per_image[0]  # shape: (num_labels,)
            probs = logits.softmax(dim=0)
            probs = probs.cpu().tolist()
        label_probs = list(zip(candidate_labels, probs))
        label_probs = sorted(label_probs, key=lambda x: x[1], reverse=True)[:top_k]
        return label_probs

    def image_to_embedding(self, image_path: str) -> torch.Tensor:
        """Return CLIP image embedding (normalized)."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.clip_device)
        with torch.no_grad():
            image_embeds = self.clip_model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds.cpu()

    def build_prompt_from_labels(self, label_probs: List[Tuple[str, float]], template: str) -> str:
        labels_str = ", ".join([f"{label} ({prob:.2f})" for label, prob in label_probs])
        return template.format(labels=labels_str)

    def generate_from_image(
    self,
    image_path: str,
    candidate_labels: List[str],
    top_k: int = 3,
    prompt_template: Optional[str] = None,
    llm_kwargs: Optional[Dict] = None,
    return_image: bool = True,
    ) -> Dict:
      """
      Main convenience method:
      - CLIP zero-shot → top_k labels
      - build prompt
      - call LLM
      Returns dict with label_probs, prompt, llm output, and optionally the PIL image.
      """
      if prompt_template is None:
          prompt_template = "Image description: {labels}. Write a detailed paragraph about the image."

      label_probs = self.image_to_top_labels(image_path, candidate_labels, top_k=top_k)
      prompt = self.build_prompt_from_labels(label_probs, prompt_template)

      llm_kwargs = llm_kwargs or {}
      outputs = self.llm(prompt, **llm_kwargs)
      generated_text = outputs[0]["generated_text"]

      result = {
          "label_probs": label_probs,
          "prompt": prompt,
          "generated_text": generated_text,
      }
      if return_image:
          from PIL import Image
          result["image"] = Image.open(image_path).convert("RGB")
      return result
