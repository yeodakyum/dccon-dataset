"""Text-to-sticker search using the two SigLIP 2 encoders."""

from itertools import islice

MODEL = "google/siglip2-base-patch16-224"
BATCH_SIZE = 16


class ImageSearch:
    def __init__(self, dataset, stickers):
        import faiss
        from PIL import Image
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL).eval()
        self.stickers = []
        self.skipped = []
        self.index = faiss.IndexFlatIP(self.model.config.vision_config.hidden_size)

        pending = iter(stickers)
        while batch := list(islice(pending, BATCH_SIZE)):
            images = []
            rows = []
            for sticker in batch:
                try:
                    path = (dataset / sticker["image_path"]).resolve()
                    if not path.is_relative_to(dataset):
                        raise ValueError("Image path is outside the dataset")
                    with Image.open(path) as source:
                        # Pillow opens animated images on their first frame.
                        foreground = source.convert("RGBA")
                        background = Image.new("RGBA", source.size, "white")
                        images.append(Image.alpha_composite(background, foreground).convert("RGB"))
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    self.skipped.append({"sticker_id": sticker["sticker_id"], "error": str(exc)})
                    continue
                rows.append(sticker)
            if images:
                inputs = self.processor(images=images, return_tensors="pt")
                self.index.add(self.embeddings(self.model.get_image_features, inputs))
                self.stickers.extend(rows)

        if not self.stickers:
            raise ValueError(f"No decodable sticker images in {dataset}")

    @staticmethod
    def embeddings(encode, inputs):
        import torch

        with torch.inference_mode():
            output = encode(**inputs)
            vectors = output if isinstance(output, torch.Tensor) else output.pooler_output
            return torch.nn.functional.normalize(vectors.float()).numpy()

    def search(self, queries, top_k=5):
        inputs = self.processor(
            text=queries, padding="max_length", truncation=True, return_tensors="pt"
        )
        scores, positions = self.index.search(
            self.embeddings(self.model.get_text_features, inputs),
            min(top_k, self.index.ntotal),
        )
        results = []
        for query, row_scores, row_positions in zip(queries, scores, positions):
            matches = []
            for rank, (score, position) in enumerate(zip(row_scores, row_positions), 1):
                sticker = self.stickers[position]
                matches.append({
                    "rank": rank,
                    "similarity": round(float(score), 4),
                    "package_id": sticker["package_id"],
                    "sticker_id": sticker["sticker_id"],
                    "image_path": sticker["image_path"],
                })
            results.append({"query": query, "embedding_input": query, "matches": matches})
        return results
