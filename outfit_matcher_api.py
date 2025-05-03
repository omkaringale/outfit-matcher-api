from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import os
import gdown

app = FastAPI()

# Paths
PKL_PATH = "clean_clip_embeddings.pkl"
GOOGLE_DRIVE_FILE_ID = "13UFKICfKjony83D8lqR_TSs6eWj20VxY"  # Replace this with the actual file ID of the .pkl.gz

# Download the pickle file using gdown
def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

# Step 1: Download and load pickle
if not os.path.exists(PKL_PATH):
    print("Downloading compressed pickle file from Google Drive...")
    download_from_gdrive(GOOGLE_DRIVE_FILE_ID, PKL_PATH)

# Step 2: Load DataFrame from compressed pickle
df = pd.read_pickle(PKL_PATH, compression="gzip")

# Step 3: Prepare topwear and bottomwear subsets
tops = df[df['subCategory'].str.lower() == 'topwear'].reset_index(drop=True)
bottoms = df[df['subCategory'].str.lower() == 'bottomwear'].reset_index(drop=True)

# Request schema
class OutfitRequest(BaseModel):
    top_id: int
    top_n: int = 3

@app.post("/suggest")
def suggest_outfit(req: OutfitRequest):
    top_id = req.top_id
    top_n = req.top_n

    if top_id not in tops['id'].values:
        raise HTTPException(status_code=404, detail=f"Topwear ID {top_id} not found")

    top_vec = np.array(tops[tops['id'] == top_id].iloc[0]['embedding_vector']).reshape(1, -1)
    bottom_vecs = np.stack(bottoms['embedding_vector'].to_numpy())
    similarities = cosine_similarity(top_vec, bottom_vecs)[0]

    bottom_df = bottoms.copy()
    bottom_df['similarity'] = similarities
    matched = bottom_df.sort_values(by='similarity', ascending=False).head(top_n)

    return matched[['id', 'productDisplayName', 'similarity']].to_dict(orient="records")

@app.get("/")
def read_root():
    return {"message": "Outfit Matcher API is live"}

# Run locally if needed
# if __name__ == "__main__":
#     uvicorn.run("outfit_matcher_api:app", host="127.0.0.1", port=8000, reload=True)
