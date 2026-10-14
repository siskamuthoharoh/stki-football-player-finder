
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PlayerRecommender:
    """Rekomendasi pemain berdasarkan kemiripan profil."""

    def __init__(self, df_pemain):
        self.df = df_pemain.reset_index(drop=True).copy()

        kolom_teks = [
            "main_position",
            "position",
            "foot",
            "citizenship",
            "country_of_birth",
        ]

        self.df["profile_text"] = (
            self.df[kolom_teks]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=50000
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["profile_text"]
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
        """Mengurutkan pemain berdasarkan skor kemiripan gabungan."""

        posisi_target = self.df.index[
            self.df["player_id"] == player_id
        ].tolist()

        if not posisi_target:
            return None

        indeks = posisi_target[0]

        # Komponen 1: kemiripan teks profil.
        skor_teks = cosine_similarity(
            self.tfidf_matrix[indeks],
            self.tfidf_matrix
        ).flatten()

        # Komponen 2: kedekatan usia.
        usia_target = self.df.loc[indeks, "age"]
        usia = pd.to_numeric(self.df["age"], errors="coerce")

        if pd.notna(usia_target):
            skor_usia = (1 - (usia - usia_target).abs() / 30).clip(0, 1)
            skor_usia = skor_usia.fillna(0).to_numpy()
            tersedia_usia = usia.notna().to_numpy()
        else:
            skor_usia = None
            tersedia_usia = None

        # Komponen 3: kedekatan tinggi badan.
        tinggi_target = self.df.loc[indeks, "height"]
        tinggi = pd.to_numeric(self.df["height"], errors="coerce")

        if pd.notna(tinggi_target):
            skor_tinggi = (
                1 - (tinggi - tinggi_target).abs() / 50
            ).clip(0, 1)
            skor_tinggi = skor_tinggi.fillna(0).to_numpy()
            tersedia_tinggi = tinggi.notna().to_numpy()
        else:
            skor_tinggi = None
            tersedia_tinggi = None

        # Bobot awal: teks 60%, usia 25%, tinggi 15%.
        pembilang = 0.60 * skor_teks
        penyebut = 0.60 * pd.Series(
            True, index=self.df.index
        ).to_numpy(dtype=float)

        if skor_usia is not None:
            pembilang += 0.25 * skor_usia * tersedia_usia
            penyebut += 0.25 * tersedia_usia

        if skor_tinggi is not None:
            pembilang += 0.15 * skor_tinggi * tersedia_tinggi
            penyebut += 0.15 * tersedia_tinggi

        skor_akhir = pembilang / penyebut

        # Pemain target tidak boleh direkomendasikan kepada dirinya sendiri.
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
