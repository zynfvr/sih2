from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-mpnet-base-v2", cache_folder="./hf_models", local_files_only=True)
