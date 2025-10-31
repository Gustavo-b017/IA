# knn_item_cf_surprise_gridsearch.py
# Item-based KNN CF with Surprise + GridSearchCV on MovieLens 100K.

from surprise import Dataset, KNNBaseline, KNNBasic, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
from collections import defaultdict

# 1) Load data (MovieLens 100k). First run will download it.
data = Dataset.load_builtin("ml-100k")

# 2) Quick holdout for final evaluation after model selection
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 3) Hyperparameter search (3-fold CV) on the TRAIN portion only
#    We search only ITEM-BASED (user_based=False). Try both cosine and pearson_baseline.
param_grid = {
    "k": [10,20,30,40],
    "min_k": [1, 3, 5],
    "sim_options": {
        "name": ["cosine", "pearson_baseline"],
        "min_support": [1, 5],
        "user_based": [False],  # False -> item-based
    },
    # Optional (only used by KNNBaseline): baseline estimates (ALS) for debiasing
    # "bsl_options": [
    #     {"method": "als", "n_epochs": 10, "reg_u": 15, "reg_i": 10},
    #     {"method": "als", "n_epochs": 15, "reg_u": 12, "reg_i": 8},
    # ]
}

# You can switch to KNNBasic below if you want a simpler algorithm:
AlgoClass = KNNBaseline  # or KNNBasic

gs = GridSearchCV(
    AlgoClass,
    param_grid,
    measures=["rmse", "mae"],
    cv=3,
    n_jobs=-1,
    joblib_verbose=0,
    refit=True,  # refit on full CV training folds using best params (rmse by default)
)

# Fit grid search on the original full dataset (it will internally do CV)
gs.fit(data)

print("Best RMSE:", gs.best_score["rmse"])
print("Best params (RMSE):", gs.best_params["rmse"])
print("Best MAE:", gs.best_score["mae"])
print("Best params (MAE):", gs.best_params["mae"])

# 4) Train the best model on the *trainset* (from the holdout split)
best_params = gs.best_params["rmse"]
algo = AlgoClass(**best_params)
algo.fit(trainset)

# 5) Evaluate on testset (holdout)
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions, verbose=True)
mae = accuracy.mae(predictions, verbose=True)

# 6) Produce Top-N recommendations for a given user (by raw user id, as string/int)
def get_top_n_for_user(algo, trainset, raw_uid, n=10):
    """Return top-n (item_raw_id, estimated_rating) for items the user hasn't rated."""
    # Surprise uses inner ids internally; map raw -> inner
    try:
        inner_uid = trainset.to_inner_uid(str(raw_uid))
    except ValueError:
        raise ValueError(f"User {raw_uid} not found in training set.")

    # Items the user has already interacted with
    items_rated_by_user = set(j for (j, _) in trainset.ur[inner_uid])

    # Iterate over all items; score only those not yet rated
    candidates = []
    for inner_iid in range(trainset.n_items):
        if inner_iid in items_rated_by_user:
            continue
        raw_iid = trainset.to_raw_iid(inner_iid)
        # Predict on-the-fly; Surprise takes (raw_uid, raw_iid)
        est = algo.predict(str(raw_uid), raw_iid, verbose=False).est
        candidates.append((raw_iid, est))

    # Sort by estimated rating descending and return top-n
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]

# Example: recommendations for user "196" (a common user id in ML-100K)
try:
    topn = get_top_n_for_user(algo, trainset, raw_uid="934", n=10)
    print("\nTop-10 recommendations for user 934:")
    for iid, est in topn:
        print(f"  MovieID={iid}  |  PredRating={est:.3f}")
except ValueError as e:
    print(e)

# 7) (Optional) Cross-validate the final AlgoClass with the found best_params on full data
#    This shows more stable metrics across folds.
final_algo = AlgoClass(**best_params)
cv_results = cross_validate(final_algo, data, measures=["rmse", "mae"], cv=5, verbose=False, n_jobs=-1)
print(
    "\n5-fold CV (refit with best params) -> RMSE mean: {:.4f} ± {:.4f}, MAE mean: {:.4f} ± {:.4f}".format(
        cv_results["test_rmse"].mean(), cv_results["test_rmse"].std(),
        cv_results["test_mae"].mean(), cv_results["test_mae"].std()
    )
)
############

import pandas as pd
import os

# ---- CONFIG ----
DATA_PATH = "/root/.surprise_data/ml-100k/ml-100k"   #  adjust if needed
USER_ID = 934                                        # target user
TOPN = 10                                            # how many recommendations to show

# ---- LOAD MOVIELENS DATA ----
ratings = pd.read_csv(
    os.path.join(DATA_PATH, "u.data"),
    sep="\t",
    header=None,
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    os.path.join(DATA_PATH, "u.item"),
    sep="|",
    header=None,
    encoding="ISO-8859-1",
    names=[
        "movie_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ],
    usecols=["movie_id", "title"]
)

# ---- USER 196'S EXISTING RATINGS ----
user_ratings = (
    ratings[ratings["user_id"] == USER_ID]
    .merge(movies, on="movie_id", how="left")
    .sort_values("rating", ascending=False)
)
print(f"\n🎬 User {USER_ID} has rated {len(user_ratings)} movies — top of their list:")
for _, row in user_ratings.head(20).iterrows():
    print(f"  ⭐ {row['title']}  |  Rating = {row['rating']}")

# ---- RECOMMENDATIONS FROM TRAINED KNN MODEL ----
# assumes 'algo' (trained Surprise model) and 'trainset' already exist in memory
def get_top_n_for_user(algo, trainset, raw_uid, n=10):
    """Return top-n (item_raw_id, estimated_rating) for items the user hasn't rated."""
    inner_uid = trainset.to_inner_uid(str(raw_uid))
    items_rated_by_user = set(j for (j, _) in trainset.ur[inner_uid])
    candidates = []
    for inner_iid in range(trainset.n_items):
        if inner_iid in items_rated_by_user:
            continue
        raw_iid = trainset.to_raw_iid(inner_iid)
        est = algo.predict(str(raw_uid), raw_iid, verbose=False).est
        candidates.append((int(raw_iid), float(est)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]

# Generate Top-N predictions
topn = get_top_n_for_user(algo, trainset, raw_uid=USER_ID, n=TOPN)

# Merge with movie titles
topn_df = pd.DataFrame(topn, columns=["movie_id", "pred_rating"]).merge(movies, on="movie_id", how="left")

print(f"\n🍿 Top-{TOPN} recommendations for user {USER_ID} based on KNN predictions:")
for _, row in topn_df.iterrows():
    print(f"  🎥 {row['title']}  |  Predicted Rating = {row['pred_rating']:.3f}")


