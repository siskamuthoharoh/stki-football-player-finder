
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PlayerRecommender:
    """Rekomendasi pemain berdasarkan kemiripan profil."""

    def __init__(self, df_pemain):
        self.df = df_pemain.reset_index(drop=True).copy()

        # Teks posisi dipisahkan agar bobotnya bisa diatur.
        self.df["position_text"] = (
            self.df[["main_position", "position"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )

        # Fitur pendukung selain posisi dan klub.
        kolom_pendukung = [
            "foot",
            "citizenship",
            "country_of_birth",
        ]

        self.df["support_text"] = (
            self.df[kolom_pendukung]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )

        self.position_vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=10000
        )
        self.position_matrix = self.position_vectorizer.fit_transform(
            self.df["position_text"]
        )

        self.support_vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=10000
        )
        self.support_matrix = self.support_vectorizer.fit_transform(
            self.df["support_text"]
        )

    def search_players(self, nama_pemain, batas_hasil=10):
        """Mencari pemain berdasarkan sebagian nama."""

        hasil = self.df[
            self.df["player_name"].str.contains(
                nama_pemain,
                case=False,
                na=False,
                regex=False
            )
        ]

        kolom_tampilan = [
            "player_id",
            "player_name",
            "main_position",
            "position",
            "current_club_name",
            "age",
            "height"
        ]

        return hasil[kolom_tampilan].head(batas_hasil).copy()

    def recommend_by_id(self, player_id, jumlah_rekomendasi=5):
        """Mengurutkan pemain berdasarkan kemiripan profil."""

        posisi_target = self.df.index[
            self.df["player_id"] == player_id
        ].tolist()

        if not posisi_target:
            return None

        indeks = posisi_target[0]

        # 1. Kemiripan posisi.
        skor_posisi = cosine_similarity(
            self.position_matrix[indeks],
            self.position_matrix
        ).flatten()

        # 2. Kemiripan fitur pendukung.
        skor_pendukung = cosine_similarity(
            self.support_matrix[indeks],
            self.support_matrix
        ).flatten()

        # 3. Kedekatan usia.
        usia_target = pd.to_numeric(
            pd.Series([self.df.loc[indeks, "age"]]),
            errors="coerce"
        ).iloc[0]

        usia = pd.to_numeric(self.df["age"], errors="coerce")

        if pd.notna(usia_target):
            skor_usia = (
                1 - (usia - usia_target).abs() / 30
            ).clip(0, 1).fillna(0).to_numpy()
            tersedia_usia = usia.notna().to_numpy()
        else:
            skor_usia = None
            tersedia_usia = None

        # 4. Kedekatan tinggi badan.
        tinggi_target = pd.to_numeric(
            pd.Series([self.df.loc[indeks, "height"]]),
            errors="coerce"
        ).iloc[0]

        tinggi = pd.to_numeric(self.df["height"], errors="coerce")

        if pd.notna(tinggi_target):
            skor_tinggi = (
                1 - (tinggi - tinggi_target).abs() / 50
            ).clip(0, 1).fillna(0).to_numpy()
            tersedia_tinggi = tinggi.notna().to_numpy()
        else:
            skor_tinggi = None
            tersedia_tinggi = None

        # Bobot dasar: posisi 50%, pendukung 10%,
        # usia 25%, dan tinggi badan 15%.
        pembilang = (
            0.50 * skor_posisi
            + 0.10 * skor_pendukung
        )
        penyebut = 0.60

        if skor_usia is not None:
            pembilang += 0.25 * skor_usia * tersedia_usia
            penyebut = penyebut + 0.25 * tersedia_usia

        if skor_tinggi is not None:
            pembilang += 0.15 * skor_tinggi * tersedia_tinggi
            penyebut = penyebut + 0.15 * tersedia_tinggi

        skor_akhir = pembilang / penyebut

        # Pemain target tidak boleh merekomendasikan dirinya sendiri.
        skor_akhir[indeks] = -1

        urutan = skor_akhir.argsort()[::-1]
        urutan = urutan[:jumlah_rekomendasi]

        kolom_tampilan = [
            "player_id",
            "player_name",
            "main_position",
            "position",
            "current_club_name",
            "age",
            "height"
        ]

        hasil = self.df.iloc[urutan][kolom_tampilan].copy()
        hasil["similarity_score"] = skor_akhir[urutan]

        return hasil.reset_index(drop=True)
