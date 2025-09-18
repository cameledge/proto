from transformers import pipeline

def build_llm(model_name="gpt2", device=-1):
    return pipeline("text-generation", model=model_name, device=device)

if __name__ == "__main__":
    llm = build_llm("gpt2")
    out = llm("The future of computer vision is", max_length=30)
    print(out[0]["generated_text"])
