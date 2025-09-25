from transformers import pipeline
from pathlib import Path


def run_sample_inference():
    """Download/cache the Granite-Docling model and run a small image+text example."""
    model_id = "ibm-granite/granite-docling-258M"
    cache_dir = Path("./models")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline (this may download weights the first time)...")
    pipe = pipeline(
        task="image-text-to-text",
        model=model_id,
        device="cpu",
        model_kwargs={
            # Ensure CPU-friendly dtype; override if you have GPU/CUDA available
            "torch_dtype": "float32",
            "cache_dir": str(cache_dir),
        },
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                },
                {"type": "text", "text": "What animal is on the candy?"},
            ],
        }
    ]

    print("Running inference...")
    # The image-text-to-text pipeline expects the conversation-style messages list
    out = pipe(messages)
    print("Response:")
    print(out)


if __name__ == "__main__":
    run_sample_inference()