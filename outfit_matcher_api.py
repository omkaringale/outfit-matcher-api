from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import ast
import os
import requests

app = FastAPI()

# Step 1: Define Google Drive File ID and destination path
CSV_PATH = "clip_embeddings.csv"
DRIVE_FILE_ID = "1Zi4fVw24OWd9Wg5g1hAwwHRRT8pk_KfI"  # <- Replace this with actual ID

# Step 2: Download the file from Google Drive if not already present
def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {dest_path} from Google Drive.")
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

if not os.path.exists(CSV_PATH):
    print("Downloading embeddings from Google Drive...")
    download_from_gdrive(DRIVE_FILE_ID, CSV_PATH)
    
df = pd.read_csv(
    CSV_PATH,
    on_bad_lines='warn',     # Log and skip problematic rows
    quotechar='"',
    escapechar='\\',
    sep=',',
    engine='python'
)

# Convert the stringified vectors to actual lists if needed
if 'embedding_vector' in df.columns:
    df['embedding_vector'] = df['embedding_vector'].apply(ast.literal_eval)
else:
# Filter out only float-type columns (512 embedding dims)
    embedding_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    df['embedding_vector'] = df[embedding_cols].values.tolist()

# Filter categories
df.columns = df.columns.str.strip()  # Clean up
if 'subCategory' not in df.columns:
    raise Exception(f"Available columns: {df.columns.tolist()}")
tops = df[df['subCategory'].str.lower() == 'topwear'].reset_index(drop=True)
bottoms = df[df['subCategory'].str.lower() == 'bottomwear'].reset_index(drop=True)

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

    bottom_vecs = np.array(bottoms['embedding_vector'].tolist())
    similarities = cosine_similarity(top_vec, bottom_vecs)[0]

    bottom_df = bottoms.copy()
    bottom_df['similarity'] = similarities
    matched = bottom_df.sort_values(by='similarity', ascending=False).head(top_n)

    return matched[['id', 'productDisplayName', 'similarity']].to_dict(orient="records")

# Uncomment below to run locally
#if __name__ == "__main__":
#    uvicorn.run("outfit_matcher_api:app", host="127.0.0.1", port=8000, reload=True)
