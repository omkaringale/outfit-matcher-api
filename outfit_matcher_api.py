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
GOOGLE_DRIVE_FILE_ID = "1MVuSmwV6G9GMKfjL0h_2hcAhSIwhvHK5"  # Replace this with the actual file ID of the .pkl.gz

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

    # Ensure valid ID
    if top_id not in tops['id'].values:
        raise HTTPException(status_code=404, detail=f"Topwear ID {top_id} not found")

    # Extract topwear vector
    top_row = tops[tops['id'] == top_id].iloc[0]
    top_vec = np.array(top_row['embedding_vector']).reshape(1, -1)

    # Optional: filter bottoms by baseColour
    color = top_row['baseColour']
    filtered_bottoms = bottoms[bottoms['baseColour'] == color]

    # Fallback if no bottoms of same color found
    if filtered_bottoms.empty:
        filtered_bottoms = bottoms.copy()

    # Compute cosine similarity
    bottom_vecs = np.stack(filtered_bottoms['embedding_vector'].to_numpy())
    similarities = cosine_similarity(top_vec, bottom_vecs)[0]

    # Attach similarities and sample randomly from top 10
    filtered_bottoms = filtered_bottoms.copy()
    filtered_bottoms['similarity'] = similarities
    top_matches = filtered_bottoms.sort_values(by='similarity', ascending=False).head(10)

    if len(top_matches) < top_n:
        matched = top_matches
    else:
        matched = top_matches.sample(n=top_n)

    # Optional: Log for debugging
    print("Top 10 similarities (id, score):", list(zip(top_matches['id'], top_matches['similarity'])))

    return matched[['id', 'productDisplayName', 'similarity']].to_dict(orient="records")

@app.get("/")
def read_root():
    return {"message": "Outfit Matcher API is live"}

# Run locally if needed
# if __name__ == "__main__":
#     uvicorn.run("outfit_matcher_api:app", host="127.0.0.1", port=8000, reload=True)
