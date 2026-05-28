import os
import re
import json
import pandas as pd

class DatasetLinker:
    """
    A utility class to link the MovieLens 100k dataset with the IMDb (aclImdb) review dataset.
    
    Features:
    - Precomputes a high-quality title-based mapping from MovieLens item_id to IMDb ID (tt#######).
    - Caches the mapping to a local JSON file to make subsequent initializations instant.
    - Indexes all 50,000 IMDb reviews from aclImdb in under 0.5 seconds.
    - Provides a clean API to retrieve review text, rating, and sentiment for any MovieLens movie.
    - Includes a robust fallback mechanism for movies without reviews in aclImdb.
    """
    
    def __init__(self, data_dir=None, cache_file="movielens_imdb_mapping.json", force_recompute=False):
        # Determine paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir or os.path.join(self.base_dir, "data")
        self.cache_path = os.path.join(self.data_dir, cache_file)
        
        self.mapping = {}       # MovieLens item_id (str/int) -> IMDb ID (str 'tt#######')
        self.review_index = {}  # IMDb ID (str) -> list of dicts: {'filepath': str, 'polarity': str, 'rating': int}
        
        # Load/Compute the mapping
        if not force_recompute and os.path.exists(self.cache_path):
            self._load_cached_mapping()
        else:
            self._compute_mapping()
            self._save_mapping_cache()
            
        # Index the reviews
        self._index_reviews()
        
    def _load_cached_mapping(self):
        print(f"Loading cached MovieLens-IMDb mappings from {self.cache_path}...")
        with open(self.cache_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
            # JSON keys are always strings, let's cast keys to int for MovieLens item_id
            self.mapping = {int(k): v for k, v in raw_map.items()}
        print(f"Loaded {len(self.mapping)} mapped movies from cache.")
        
    def _save_mapping_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, indent=4)
        print(f"Saved MovieLens-IMDb mappings to cache: {self.cache_path}")
        
    def _compute_mapping(self):
        print("Computing new MovieLens 100k -> IMDb ID mapping...")
        
        ml_item_path = os.path.join(self.data_dir, "ml-100k", "u.item")
        small_movies_path = os.path.join(self.data_dir, "ml-latest-small", "movies.csv")
        small_links_path = os.path.join(self.data_dir, "ml-latest-small", "links.csv")
        
        if not os.path.exists(ml_item_path):
            raise FileNotFoundError(f"MovieLens 100k u.item not found at {ml_item_path}. Run download_datasets.py first.")
            
        if not os.path.exists(small_movies_path) or not os.path.exists(small_links_path):
            raise FileNotFoundError(
                f"ml-latest-small dataset files not found under {self.data_dir}. "
                "Ensure ml-latest-small.zip was downloaded and extracted."
            )
            
        # Load u.item
        u_cols = ["item_id", "title", "release_date", "video_date", "imdb_url"] + [str(i) for i in range(19)]
        u_item = pd.read_csv(ml_item_path, sep="|", encoding="latin-1", names=u_cols, header=None)
        
        # Load ml-latest-small movies & links
        movies = pd.read_csv(small_movies_path)
        links = pd.read_csv(small_links_path)
        small_df = pd.merge(movies, links, on="movieId")
        
        # Extract years
        def extract_year(title):
            match = re.search(r"\((\d{4})\)", title)
            return int(match.group(1)) if match else None

        u_item["year"] = u_item["title"].apply(extract_year)
        small_df["year"] = small_df["title"].apply(extract_year)
        
        # Text normalization helper
        def normalize_title(title):
            if not isinstance(title, str):
                return ""
            title = title.replace("&", "and")
            title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
            title = title.lower().strip()
            title = re.sub(r",\s*(the|a|an|of|in|for|and|or|on|at|by|with|to)\b", "", title)
            title = re.sub(r"[^a-z0-9\s]", "", title)
            title = re.sub(r"\bse7en\b", "seven", title)
            title = re.sub(r"\btwelve\b", "12", title)
            return title.strip()

        u_item["clean_title"] = u_item["title"].apply(normalize_title)
        small_df["clean_title"] = small_df["title"].apply(normalize_title)
        
        temp_mapping = {}
        
        # Step 1: Exact Clean Title + Year match
        small_by_clean_year = {
            (row["clean_title"], row["year"]): row["imdbId"] 
            for _, row in small_df.iterrows() 
            if row["clean_title"] and row["year"]
        }
        for _, row in u_item.iterrows():
            m_id = row["item_id"]
            key = (row["clean_title"], row["year"])
            if key in small_by_clean_year:
                temp_mapping[m_id] = small_by_clean_year[key]
                
        # Step 2: Substring Prefix + Year match (handling suffix differences)
        unmatched_u_item = u_item[~u_item["item_id"].isin(temp_mapping)]
        small_by_year = {}
        for _, row in small_df.iterrows():
            if row["year"]:
                small_by_year.setdefault(row["year"], []).append((row["clean_title"], row["imdbId"]))
                
        for _, row in unmatched_u_item.iterrows():
            m_id = row["item_id"]
            y = row["year"]
            c_t = row["clean_title"]
            if not y or not c_t:
                continue
            candidates = small_by_year.get(y, [])
            best_candidate = None
            for cand_ct, cand_imdb in candidates:
                if c_t in cand_ct or cand_ct in c_t:
                    best_candidate = cand_imdb
                    break
            if best_candidate:
                temp_mapping[m_id] = best_candidate
                
        # Step 3: Clean Title match without year constraint (handling year drift)
        unmatched_u_item = u_item[~u_item["item_id"].isin(temp_mapping)]
        small_by_clean = {
            row["clean_title"]: row["imdbId"] 
            for _, row in small_df.iterrows() 
            if row["clean_title"]
        }
        for _, row in unmatched_u_item.iterrows():
            m_id = row["item_id"]
            c_t = row["clean_title"]
            if c_t in small_by_clean:
                temp_mapping[m_id] = small_by_clean[c_t]
                
        # Step 4: Substring match without year constraint (highest recall fallback)
        unmatched_u_item = u_item[~u_item["item_id"].isin(temp_mapping)]
        small_list = [(row["clean_title"], row["imdbId"]) for _, row in small_df.iterrows() if row["clean_title"]]
        for _, row in unmatched_u_item.iterrows():
            m_id = row["item_id"]
            c_t = row["clean_title"]
            if not c_t or len(c_t) <= 4:
                continue
            best_cand = None
            for cand_ct, cand_imdb in small_list:
                if c_t in cand_ct or cand_ct in c_t:
                    best_cand = cand_imdb
                    break
            if best_cand:
                temp_mapping[m_id] = best_cand
                
        # Format mapping values as standard IMDb tt####### string keys
        self.mapping = {}
        for m_id, imdb_num in temp_mapping.items():
            if pd.notna(imdb_num):
                self.mapping[m_id] = f"tt{int(imdb_num):07d}"
                
        print(f"Mapping computation finished. Successfully matched {len(self.mapping)} / {len(u_item)} movies.")
        
    def _index_reviews(self):
        print("Indexing aclImdb reviews...")
        start_time = pd.Timestamp.now()
        
        self.review_index = {}
        subsets = [
            ("train", "pos"),
            ("train", "neg"),
            ("test", "pos"),
            ("test", "neg")
        ]
        
        for set_name, polarity in subsets:
            urls_path = os.path.join(self.data_dir, "aclImdb", set_name, f"urls_{polarity}.txt")
            folder_path = os.path.join(self.data_dir, "aclImdb", set_name, polarity)
            
            if not os.path.exists(urls_path) or not os.path.exists(folder_path):
                continue
                
            with open(urls_path, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f]
                
            files = os.listdir(folder_path)
            for filename in files:
                if filename.endswith(".txt"):
                    match = re.match(r"(\d+)_(\d+)\.txt", filename)
                    if match:
                        idx = int(match.group(1))
                        rating = int(match.group(2))
                        
                        if idx < len(urls):
                            url = urls[idx]
                            tt_match = re.search(r"tt\d+", url)
                            if tt_match:
                                tt_id = tt_match.group(0)
                                filepath = os.path.join(folder_path, filename)
                                self.review_index.setdefault(tt_id, []).append({
                                    "filepath": filepath,
                                    "polarity": polarity,
                                    "rating": rating
                                })
                                
        total_reviews = sum(len(v) for v in self.review_index.values())
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        print(f"Indexed {total_reviews:,} reviews across {len(self.review_index):,} IMDb movies in {elapsed:.3f} seconds.")
        
    def get_imdb_id(self, item_id):
        """
        Returns the IMDb ID (tt#######) for a given MovieLens item_id.
        """
        return self.mapping.get(int(item_id))
        
    def get_reviews(self, item_id):
        """
        Fetches all review objects available for a MovieLens item_id.
        
        Returns:
            list of dicts: [
                {
                    'text': str (contents of review file),
                    'polarity': 'pos' | 'neg',
                    'rating': int (1-10)
                }, ...
            ]
        """
        imdb_id = self.get_imdb_id(item_id)
        if not imdb_id or imdb_id not in self.review_index:
            return []
            
        reviews = []
        for review_meta in self.review_index[imdb_id]:
            try:
                with open(review_meta["filepath"], "r", encoding="utf-8") as f:
                    text = f.read()
                reviews.append({
                    "text": text,
                    "polarity": review_meta["polarity"],
                    "rating": review_meta["rating"]
                })
            except Exception as e:
                # Handle potential read errors gracefully
                continue
                
        return reviews
        
    def get_sentiment_score(self, item_id, default_score=0.5):
        """
        Calculates the average review rating normalized to a [0.0, 1.0] scale,
        or uses the binary polarity count if rating isn't ideal.
        
        If no reviews are available in aclImdb, returns the default_score.
        """
        reviews = self.get_reviews(item_id)
        if not reviews:
            return default_score
            
        # Method A: Normalized star ratings (1-10 mapped to [0, 1])
        ratings = [r["rating"] for r in reviews if r["rating"] is not None]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            # Map 1-10 scale to 0.0-1.0
            return (avg_rating - 1.0) / 9.0
            
        # Method B: Polarity binary mapping
        polarities = [1.0 if r["polarity"] == "pos" else 0.0 for r in reviews]
        return sum(polarities) / len(polarities)

# self-test code when run directly
if __name__ == "__main__":
    linker = DatasetLinker()
    
    # Test on Toy Story (item_id = 1)
    toy_story_id = 1
    imdb_id = linker.get_imdb_id(toy_story_id)
    reviews = linker.get_reviews(toy_story_id)
    sentiment = linker.get_sentiment_score(toy_story_id)
    
    print("\n" + "=" * 50)
    print("Dataset Linker Self-Test Output")
    print("=" * 50)
    print(f"MovieLens item_id: {toy_story_id}")
    print(f"IMDb ID: {imdb_id}")
    print(f"Number of reviews found: {len(reviews)}")
    print(f"Computed Sentiment score: {sentiment:.4f}")
    
    if reviews:
        print("\nSample Review Snippet:")
        snippet = reviews[0]["text"][:250] + "..."
        print(f"Rating: {reviews[0]['rating']}/10 | Polarity: {reviews[0]['polarity']}")
        print(f"Text:\n{snippet}")
    print("=" * 50)
