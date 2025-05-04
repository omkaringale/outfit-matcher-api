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
GOOGLE_DRIVE_FILE_ID = "13UFKICfKjony83D8lqR_TSs6eWj20VxY"  # Replace with actual ID

# Download from Google Drive if needed
def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

if not os.path.exists(PKL_PATH):
    print("Downloading compressed pickle file from Google Drive...")
    download_from_gdrive(GOOGLE_DRIVE_FILE_ID, PKL_PATH)

# Load DataFrame
df = pd.read_pickle(PKL_PATH, compression="gzip")

# Function to infer fashion style
def extract_style(name):
    if not isinstance(name, str):
        return 'casual'  # Default
    name = name.lower()
    if 'jeans' in name or 'denim' in name:
        return 'casual'
    elif 'formal' in name or 'trousers' in name:
        return 'formal'
    elif 'jogger' in name or 'track' in name:
        return 'sports'
    else:
        return 'casual'

# Preprocess
df['style'] = df['productDisplayName'].apply(extract_style)
tops = df[df['subCategory'].str.lower() == 'topwear'].reset_index(drop=True)
bottoms = df[df['subCategory'].str.lower() == 'bottomwear'].reset_index(drop=True)

# Add style to tops and bottoms
tops['style'] = tops['productDisplayName'].apply(extract_style)
bottoms['style'] = bottoms['productDisplayName'].apply(extract_style)

# Input Schema
class OutfitRequest(BaseModel):
    top_id: int
    top_n: int = 3

@app.post("/suggest")
def suggest_outfit(req: OutfitRequest):
    top_id = req.top_id
    top_n = req.top_n

    if top_id not in tops['id'].values:
        raise HTTPException(status_code=404, detail=f"Topwear ID {top_id} not found")

    top_row = tops[tops['id'] == top_id].iloc[0]
    top_vec = np.array(top_row['embedding_vector']).reshape(1, -1)
    top_style = top_row['style']

    # Filter bottoms by matching style
    style_matched_bottoms = bottoms[bottoms['style'] == top_style].copy()
    if style_matched_bottoms.empty:
        style_matched_bottoms = bottoms.copy()  # fallback

    bottom_vecs = np.stack(style_matched_bottoms['embedding_vector'].to_numpy())
    similarities = cosine_similarity(top_vec, bottom_vecs)[0]

    style_matched_bottoms['similarity'] = similarities
    matched = style_matched_bottoms.sort_values(by='similarity', ascending=False).head(top_n)

    return matched[['id', 'productDisplayName', 'subCategory', 'style', 'similarity']].to_dict(orient="records")

@app.get("/")
def read_root():
    return {"message": "Outfit Matcher API is live"}

# Uncomment below to run locally
# if __name__ == "__main__":
#     uvicorn.run("outfit_matcher_api:app", host="127.0.0.1", port=8000, reload=True)
