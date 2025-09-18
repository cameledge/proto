# scripts/multimodal_run.py
import hydra
from omegaconf import DictConfig
from myproto.pipelines.multimodal import MultiModalPipeline

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # build pipeline from config
    mm = MultiModalPipeline(
        clip_model_name=cfg.model.clip.model_name,
        clip_device=cfg.model.clip.device,
        llm_model_name=cfg.model.llm.model_name,
        llm_device=cfg.model.llm.device,
        llm_max_length=cfg.model.llm.max_length,
        llm_do_sample=cfg.model.llm.do_sample,
    )

    # candidate labels (example). You could also load a label list from file/config.
    candidate_labels = ["a cat", "a dog", "a car", "a bicycle", "a person", "a building", "an outdoor scene", "a microscopic image"]

    out = mm.generate_from_image(
        image_path=cfg.input.image_path,
        candidate_labels=candidate_labels,
        top_k=cfg.model.prompt.top_k,
        prompt_template=cfg.model.prompt.template,
    )

    print("Top labels and probs:", out["label_probs"])
    print("\nPrompt sent to LLM:\n", out["prompt"])
    print("\nLLM output:\n", out["generated_text"])


if __name__ == "__main__":
    main()
