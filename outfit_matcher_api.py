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
GOOGLE_DRIVE_FILE_ID = "13UFKICfKjony83D8lqR_TSs6eWj20VxY"

# Download the pickle file using gdown
def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

if not os.path.exists(PKL_PATH):
    print("Downloading compressed pickle file from Google Drive...")
    download_from_gdrive(GOOGLE_DRIVE_FILE_ID, PKL_PATH)

# Load DataFrame from compressed pickle
df = pd.read_pickle(PKL_PATH, compression="gzip")

# Preprocess
def infer_style(row):
    name = row['productDisplayName'].lower()
    if "shirt" in name or "formal" in name:
        return "formal"
    elif "jeans" in name or "t-shirt" in name:
        return "casual"
    elif "kurta" in name or "ethnic" in name:
        return "ethnic"
    elif "jacket" in name:
        return "layered"
    else:
        return "general"

df['style_tag'] = df.apply(infer_style, axis=1)

# Split tops and bottoms
tops = df[df['subCategory'].str.lower() == 'topwear'].reset_index(drop=True)
bottoms = df[df['subCategory'].str.lower() == 'bottomwear'].reset_index(drop=True)

STYLE_COMPATIBILITY = {
    "formal": ["formal", "general"],
    "casual": ["casual", "general"],
    "ethnic": ["ethnic", "general"],
    "layered": ["casual", "formal", "general"],
    "general": ["formal", "casual", "ethnic", "layered", "general"]
}

# Request schema
class OutfitRequest(BaseModel):
    top_id: int
    top_n: int = 3

@app.post("/suggest")
def suggest_outfit(req: OutfitRequest):
    if req.top_id not in tops['id'].values:
        raise HTTPException(status_code=404, detail=f"Topwear ID {req.top_id} not found")

    top_row = tops[tops['id'] == req.top_id].iloc[0]
    top_vec = np.array(top_row['embedding_vector']).reshape(1, -1)
    top_style = top_row['style_tag']

    bottom_vecs = np.stack(bottoms['embedding_vector'].to_numpy())
    similarities = cosine_similarity(top_vec, bottom_vecs)[0]

    bottom_df = bottoms.copy()
    bottom_df['similarity'] = similarities

    compatible_styles = STYLE_COMPATIBILITY.get(top_style, ["general"])
    filtered = bottom_df[bottom_df['style_tag'].isin(compatible_styles)]

    matched = filtered.sort_values(by='similarity', ascending=False).head(req.top_n)

    return matched[['id', 'productDisplayName', 'baseColour', 'style_tag', 'similarity']].to_dict(orient="records")

@app.get("/")
def read_root():
    return {"message": "Outfit Matcher API is live"}

# Uncomment for local testing
# if __name__ == "__main__":
#     uvicorn.run("outfit_matcher_api:app", host="127.0.0.1", port=8000, reload=True)
