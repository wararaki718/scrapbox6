import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def main() -> None:
    cps_df = pd.read_stata("https://users.nber.org/~rdehejia/data/cps_controls.dta")
    print(f"cps: {cps_df.shape}")

    nsw_df = pd.read_stata("https://users.nber.org/~rdehejia/data/nsw_dw.dta")
    print(f"nsw: {nsw_df.shape}")

    df = pd.concat([nsw_df[nsw_df.treat == 1], cps_df], ignore_index=True)
    print(f"nsw[treat==1] + cps: {df.shape}")
    print()

    print(df.treat.value_counts())
    print()

    # data
    X_train = df.drop(["data_id", "treat", "re78"], axis=1).values
    y_train = df.treat.values
    print(f"X_train: {X_train}")
    print(f"y_train: {y_train}")

    # model training
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    print(f"X_train_std: {X_train_std.shape}")

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("model trained!")

    # prediction
    result_df = df.drop(["data_id", "re78"], axis=1)
    result_df.reset_index(inplace=True, names="user_id")
    result_df["Zscore"] = model.predict_proba(X_train)[:, 1]
    print(result_df.head(3))

    # pscore matching

    print("DONE")


if __name__ == "__main__":
    main()
