
from pathlib import Path

import pandas as pd


def load_player_data(dataset_path=None):
    """Memuat dataset pemain sepak bola."""

    if dataset_path is None:
        project_root = Path(__file__).resolve().parent.parent
        dataset_path = project_root / "dataset" / "player_profiles.csv"
    else:
        dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {dataset_path}"
        )

    return pd.read_csv(dataset_path, low_memory=False)


def preprocess_player_data(df):
    """Membersihkan data dan memilih fitur pemain."""

    df_clean = df.copy()

    df_clean["date_of_birth"] = pd.to_datetime(
        df_clean["date_of_birth"],
        errors="coerce"
    )

    tanggal_acuan = pd.Timestamp("2026-10-14")
    tanggal_lahir = df_clean["date_of_birth"]

    sudah_ulang_tahun = (
        (tanggal_lahir.dt.month < tanggal_acuan.month)
        | (
            (tanggal_lahir.dt.month == tanggal_acuan.month)
            & (tanggal_lahir.dt.day <= tanggal_acuan.day)
        )
    )

    usia = (
        tanggal_acuan.year
        - tanggal_lahir.dt.year
        - (~sudah_ulang_tahun).astype("Int64")
    )

    df_clean["age"] = usia.where(tanggal_lahir.notna())

    # Menandai tinggi badan yang tidak valid sebagai nilai kosong
    df_clean.loc[df_clean["height"] <= 2, "height"] = float("nan")

    kolom_fitur = [
    "player_id",
    "player_name",
    "main_position",
    "position",
    "current_club_name",
    "citizenship",
    "country_of_birth",
    "place_of_birth",
    "foot",
    "height",
    "age",
]

    df_pemain = df_clean[kolom_fitur].copy()

    # Hanya pemain yang memiliki nama masuk ke data pencarian
    df_pencarian = df_pemain.loc[
        df_pemain["player_name"].notna()
    ].copy()

    return df_pencarian
