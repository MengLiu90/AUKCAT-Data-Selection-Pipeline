import os
import sys
import subprocess
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# --- Config ---
RANDOM_STATE = 123
MAX_NO_IMPROVE_ROUNDS = 20
RESAMPLE_CAP = 5
N_ESTIMATORS = 100

ORIG_CSV = 'Datasets/example_data.csv'
PSEUDO_CSV = 'Synthetic_data/generated_synthetic_data_3_per_original_instance.csv'
GEN_SCRIPT = 'CVAE_data_generation.py'

OUT_DIR = f'pseudo_selection_outputs_20r_tolorance'
os.makedirs(OUT_DIR, exist_ok=True)

seed_log_path = os.path.join(OUT_DIR, "seed_log.txt")

feature_columns = [f'mol2vec_{i}' for i in range(300)] + \
                  [f'ec2vec_{i}' for i in range(1024)] + \
                  [f'Embedding_{i+1}' for i in range(128)]
target_column = 'kcat'
target_column_aug = 'kcat_cosine'

# --- Load original dataset ---
df = pd.read_csv(ORIG_CSV)
# df['original_index'] = df.index

# Baseline state
rmse_log_prev = None
df_kept_all = []
no_improve_rounds = 0
round_idx = 0
rf_best = None

while True:
    round_idx += 1
    print(f"\n========== ROUND {round_idx} ==========")
    
    split_seed = RANDOM_STATE + round_idx
    print(f"[ROUND {round_idx}] Split seed used: {split_seed}")

    # Log split seed to file
    with open(seed_log_path, "a") as f:
        f.write(f"[ROUND {round_idx}] Split seed: {split_seed}\n")


    # Split training/test differently each round
    train_org, test_org = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE + round_idx)
    train_org = train_org.reset_index(drop=True)
    test_org = test_org.reset_index(drop=True)

    X_train_org_df = train_org[feature_columns]
    y_train_org_log_s = np.log10(train_org[target_column])
    X_test = test_org[feature_columns].values
    y_test_log = np.log10(test_org[target_column].values)

    # Train initial model on current split
    rf_curr = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf_curr.fit(X_train_org_df, y_train_org_log_s)
    y_pred_log = rf_curr.predict(X_test)
    rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
    mse_log = mean_squared_error(y_test_log, y_pred_log)
    corr_log, p_value_log = pearsonr(y_test_log, y_pred_log)
    r2 = r2_score(y_test_log, y_pred_log)

    if rmse_log_prev is None:
        rf_best = rf_curr
        rmse_log_prev = rmse_log
        print(f"[ROUND {round_idx}] Initial RMSE_log = {rmse_log_prev:.6f}")
        print(f"[ROUND {round_idx}] MSE_log={mse_log:.6f}")
        print(f"[ROUND {round_idx}] Pearson's correlation coefficient on log scale={corr_log:.6f}")
        print(f"[ROUND {round_idx}] R-squared value transformed={r2:.6f}")
        
        continue

    # Generate pseudo data
    subprocess.run([sys.executable, GEN_SCRIPT], check=True)
    df_pseudo = pd.read_csv(PSEUDO_CSV)

    # Filter pseudo by current training data source only
    df_pseudo = df_pseudo[df_pseudo['source_index'].isin(train_org['original_index'])].copy()

    # Predict on pseudo data
    X_pseudo = df_pseudo[feature_columns].values
    y_pseudo_pred_log = rf_best.predict(X_pseudo)
    df_pseudo['rf_pred_log10'] = y_pseudo_pred_log
    df_pseudo['rf_pred'] = np.power(10.0, y_pseudo_pred_log)

    df_valid = df_pseudo[df_pseudo[target_column_aug] > 0].copy()
    df_valid['knn_log10'] = np.log10(df_valid[target_column_aug])
    df_valid['abs_diff_log10'] = np.abs(df_valid['rf_pred_log10'] - df_valid['knn_log10'])

    accept_mask = df_valid['abs_diff_log10'] <= rmse_log_prev
    df_accept = df_valid[accept_mask].copy()
    df_reject = df_valid[~accept_mask].copy()

    acc_path = os.path.join(OUT_DIR, f"pseudo_selected_within_rmse_logscale_round{round_idx}.csv")
    rej_path = os.path.join(OUT_DIR, f"pseudo_rejected_outside_rmse_logscale_round{round_idx}.csv")
    df_accept.to_csv(acc_path, index=False)
    df_reject.to_csv(rej_path, index=False)

    print(f"[ROUND {round_idx}] Accepted = {len(df_accept)}, Rejected = {len(df_reject)}")

    if len(df_accept) == 0:
        no_improve_rounds += 1
        if no_improve_rounds >= MAX_NO_IMPROVE_ROUNDS:
            print("Patience exhausted. Stopping.")
            break
        continue

    improved_this_round = False
    need_n = len(df)
    replace_flag_all = len(df_accept) < need_n

    for attempt in range(1, RESAMPLE_CAP + 1):
        print(f"[ROUND {round_idx}] Attempt {attempt}/{RESAMPLE_CAP}")
        
        resample_seed = RANDOM_STATE + round_idx * 100 + attempt
        print(f"    Resample seed used: {resample_seed}")

        with open(seed_log_path, "a") as f:
            f.write(f"    Attempt {attempt}/{RESAMPLE_CAP} → Resample seed: {resample_seed}\n")
            
        df_sample = df_accept.sample(n=need_n, replace=replace_flag_all,
                                     random_state=RANDOM_STATE + round_idx * 100 + attempt)

        parts_X = [X_train_org_df]
        parts_y = [y_train_org_log_s]

        if len(df_kept_all) > 0:
            df_kept_concat = pd.concat(df_kept_all, ignore_index=True)
            ## to make sure only pseudo instances from the current training set are used to train the model
            df_kept_concat = df_kept_concat[df_kept_concat['source_index'].isin(train_org['original_index'])].copy()
            parts_X.append(df_kept_concat[feature_columns])
            parts_y.append(np.log10(df_kept_concat[target_column_aug]))
        parts_X.append(df_sample[feature_columns])
        parts_y.append(np.log10(df_sample[target_column_aug]))

        X_train_curr_df = pd.concat(parts_X, ignore_index=True)
        y_train_curr_log_s = pd.concat([pd.Series(y) for y in parts_y], ignore_index=True)

        rf_tmp = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        rf_tmp.fit(X_train_curr_df, y_train_curr_log_s)
        y_pred_log_tmp = rf_tmp.predict(X_test)

        rmse_log_new = np.sqrt(mean_squared_error(y_test_log, y_pred_log_tmp))
        print(f"    RMSE_log_new = {rmse_log_new:.6f} (prev {rmse_log_prev:.6f})")

        mse_log_new = mean_squared_error(y_test_log, y_pred_log_tmp)
        corr_log_new, p_value_log_new = pearsonr(y_test_log, y_pred_log_tmp)
        r2_new = r2_score(y_test_log, y_pred_log_tmp)

        print(f"[ROUND {round_idx}] MSE_log={mse_log_new:.6f}")
        print(f"[ROUND {round_idx}] Pearson's correlation coefficient on log scale={corr_log_new:.6f}")
        print(f"[ROUND {round_idx}] R-squared value transformed={r2_new:.6f}")

        if rmse_log_new < rmse_log_prev:
            kept_path = os.path.join(OUT_DIR, f"instances_kept_round_{round_idx}.csv")
            df_kept_round = df_sample.assign(selection_round=f"round_{round_idx}")
            df_kept_round.to_csv(kept_path, index=False)
            df_kept_all.append(df_kept_round)
            rf_best = rf_tmp
            rmse_log_prev = rmse_log_new
            improved_this_round = True
            no_improve_rounds = 0
            print(f"    Improvement accepted → {kept_path}")
            break

    if not improved_this_round:
        no_improve_rounds += 1
        if no_improve_rounds >= MAX_NO_IMPROVE_ROUNDS:
            print("Patience exhausted. Stopping.")
            break

print("Done.")
