from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import fpgrowth, association_rules
import base64
from io import BytesIO
import logging
from sklearn.decomposition import PCA  # Import PCA

# Konfigurasi logging agar pesan print terlihat di konsol Flask
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)


# ===== PREPROCESSING =====
def load_data():
    """
    Memuat data dari file CSV, melakukan pembersihan data dasar,
    dan mengambil sampel acak untuk analisis.
    Meningkatkan ukuran sampel untuk meningkatkan peluang rekomendasi dan clustering.
    """
    try:
        df = pd.read_csv('dataset/Reviews.csv')
    except FileNotFoundError:
        logging.error(
            "Error: File 'dataset/Reviews.csv' not found. Please ensure the file is in the correct directory.")
        return pd.DataFrame()  # Return empty DataFrame if file not found

    df = df.drop_duplicates()
    # Pastikan kolom yang dibutuhkan ada sebelum dropna
    required_columns = ['UserId', 'ProductId', 'Score']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"Error: Missing required columns. Found: {df.columns.tolist()}, Required: {required_columns}")
        return pd.DataFrame()  # Return empty DataFrame if columns are missing

    df = df.dropna(subset=required_columns)
    df = df[required_columns]
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df = df.dropna(subset=['Score'])  # Drop rows where score conversion failed

    # Mengambil sampel data yang lebih besar untuk meningkatkan peluang rekomendasi
    # Mengambil 20000 sampel, jika data asli kurang dari 20000, ambil semua
    sample_size = min(20000, len(df))
    if sample_size == 0:
        logging.warning("Warning: No data available after preprocessing to sample.")
        return pd.DataFrame()

    logging.info(f"Sampling {sample_size} rows from the dataset.")
    return df.sample(sample_size, random_state=42)


# ===== EDA =====
def generate_visualizations(df):
    """
    Menghasilkan visualisasi data dan menyimpannya sebagai gambar
    yang dienkode base64 untuk ditampilkan di HTML.
    Ukuran figure diperbesar untuk tampilan yang lebih jelas.
    """
    visualizations = {}

    if df.empty:
        logging.warning("DataFrame is empty, skipping visualizations.")
        return visualizations

    # Distribusi Rating
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Score', data=df)
    plt.title('Distribusi Rating', fontsize=16)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Jumlah', fontsize=12)
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    visualizations['rating_dist'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    logging.info("Generated Rating Distribution visualization.")

    # Produk Paling Populer
    if not df['ProductId'].empty:
        top_products = df['ProductId'].value_counts().head(10)
        if not top_products.empty:
            plt.figure(figsize=(12, 7))
            sns.barplot(x=top_products.index, y=top_products.values, palette='viridis', hue=top_products.index,
                        legend=False)
            plt.title('10 Produk Paling Populer', fontsize=16)
            plt.xlabel('ID Produk', fontsize=12)
            plt.ylabel('Jumlah Review', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            visualizations['top_products'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            logging.info("Generated Top Products visualization.")
        else:
            logging.warning("No top products to visualize after value_counts.")
    else:
        logging.warning("ProductId column is empty, skipping Top Products visualization.")

    # Reviewer Teraktif
    if not df['UserId'].empty:
        top_reviewers = df['UserId'].value_counts().head(10)
        if not top_reviewers.empty:
            plt.figure(figsize=(12, 7))
            sns.barplot(x=top_reviewers.index, y=top_reviewers.values, palette='magma', hue=top_reviewers.index,
                        legend=False)
            plt.title('10 Reviewer Teraktif', fontsize=16)
            plt.xlabel('ID Pengguna', fontsize=12)
            plt.ylabel('Jumlah Review', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            visualizations['top_reviewers'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            logging.info("Generated Top Reviewers visualization.")
        else:
            logging.warning("No top reviewers to visualize after value_counts.")
    else:
        logging.warning("UserId column is empty, skipping Top Reviewers visualization.")

    return visualizations


# ===== CLUSTERING =====
def cluster_users(df, n_clusters=3):
    """
    Melakukan clustering pengguna berdasarkan rating produk.
    Menghasilkan data frame dengan kolom cluster dan visualisasi cluster menggunakan PCA.
    """
    clustered_users_df = pd.DataFrame()
    cluster_viz = None

    if df.empty:
        logging.warning("DataFrame is empty, skipping user clustering.")
        return clustered_users_df, cluster_viz

    # Buat matriks pivot UserId vs ProductId
    user_product_matrix = df.pivot_table(index='UserId', columns='ProductId', values='Score').fillna(0)
    logging.info(f"Ukuran user_product_matrix: {user_product_matrix.shape}")
    logging.info(f"Jumlah kolom user_product_matrix (produk unik): {user_product_matrix.shape[1]}")

    # Pastikan ada cukup sampel untuk clustering
    if user_product_matrix.shape[0] < n_clusters:
        logging.warning(
            f"Jumlah pengguna unik ({user_product_matrix.shape[0]}) kurang dari jumlah cluster yang diminta ({n_clusters}). "
            "Mengurangi jumlah cluster atau tidak melakukan clustering.")
        if user_product_matrix.shape[0] > 0:
            actual_n_clusters = user_product_matrix.shape[0]
            logging.info(f"Menggunakan {actual_n_clusters} cluster.")
        else:
            logging.warning("Tidak ada data pengguna unik untuk clustering.")
            return clustered_users_df, cluster_viz
    else:
        actual_n_clusters = n_clusters

    # Pastikan ada cukup fitur (kolom) untuk PCA dan clustering
    # Minimal 2 fitur untuk visualisasi PCA 2D
    if user_product_matrix.shape[1] < 2:
        logging.warning(
            "Jumlah fitur (produk unik) untuk clustering kurang dari 2. PCA dan visualisasi cluster tidak dapat dibuat.")
        # Lanjutkan clustering tanpa PCA jika fiturnya terlalu sedikit
        try:
            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
            user_product_matrix['Cluster'] = kmeans.fit_predict(user_product_matrix)
            clustered_users_df = user_product_matrix
            logging.info("KMeans clustering completed without PCA.")
        except Exception as e:
            logging.error(f"Error during KMeans clustering without PCA: {e}")
        return clustered_users_df, cluster_viz  # cluster_viz akan tetap None

    # Lakukan PCA untuk reduksi dimensi sebelum clustering
    # Reduksi ke 2 komponen untuk visualisasi
    try:
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(user_product_matrix)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=user_product_matrix.index)
        logging.info("PCA completed. Reduced to 2 components.")
    except Exception as e:
        logging.error(f"Error during PCA: {e}. Perhaps not enough data points or features? {user_product_matrix.shape}")
        return clustered_users_df, cluster_viz

    # Lakukan KMeans Clustering pada komponen PCA
    try:
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        pca_df['Cluster'] = kmeans.fit_predict(pca_df)

        # Gabungkan kembali informasi cluster ke matriks asli atau DataFrame yang relevan
        clustered_users_df = user_product_matrix.copy()
        clustered_users_df['Cluster'] = pca_df['Cluster']
        logging.info("KMeans clustering completed on PCA components.")

        # Visualisasi Cluster menggunakan komponen PCA
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df,
                        palette='viridis', s=100, alpha=0.7)
        plt.title('Visualisasi Cluster Pengguna (dengan PCA)', fontsize=16)
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.2f}% Variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.2f}% Variance)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        cluster_viz = base64.b64encode(buffer.getvalue()).decode('utf-8')
        logging.info("Generated User Cluster visualization using PCA.")

    except Exception as e:
        logging.error(f"Error during KMeans clustering or visualization: {e}")

    return clustered_users_df, cluster_viz


# ===== REKOMENDASI (menggunakan Association Rules) =====
def generate_recommendations(df):
    """
    Menghasilkan aturan asosiasi untuk rekomendasi produk menggunakan FP-Growth.
    """
    rules = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

    if df.empty:
        logging.warning("DataFrame is empty, skipping recommendation generation.")
        return rules

    # Ubah data ke format transaksi (basket format)
    transactions_df = df.groupby('UserId')['ProductId'].apply(list).reset_index()
    transactions = transactions_df['ProductId'].tolist()
    logging.info(f"Jumlah transaksi unik yang terbentuk: {len(transactions)}")

    if not transactions:
        logging.warning("Peringatan: Tidak ada transaksi untuk dianalisis.")
        return rules

    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()

    try:
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        logging.info(f"DataFrame after one-hot encoding shape: {df_encoded.shape}")
        logging.info(f"Jumlah item unik di transaksi: {len(te.columns_)}")
    except ValueError as e:
        logging.error(
            f"Error saat TransactionEncoder: {e}. Mungkin tidak ada transaksi valid atau itemset terlalu besar.")
        return rules

    if df_encoded.empty:
        logging.warning("Peringatan: DataFrame yang dienkode kosong. Tidak dapat menghitung itemset yang sering.")
        return rules

    # Lakukan FP-Growth
    # min_support disesuaikan ke nilai yang lebih rendah lagi untuk menemukan lebih banyak aturan
    # Ini adalah parameter kunci yang perlu disesuaikan dengan karakteristik dataset Anda.
    min_support_value = 0.0001  # Menurunkan min_support lebih jauh
    logging.info(f"Mencoba FP-Growth dengan min_support={min_support_value}")

    try:
        frequent_itemsets = fpgrowth(df_encoded, min_support=min_support_value, use_colnames=True)
        logging.info(f"Jumlah frequent itemsets yang ditemukan: {len(frequent_itemsets)}")
    except Exception as e:
        logging.error(f"Error during fpgrowth: {e}")
        return rules

    if frequent_itemsets.empty:
        logging.warning(
            f"Peringatan: frequent_itemsets kosong dengan min_support={min_support_value}. Coba turunkan min_support lagi atau tingkatkan ukuran dataset."
        )
        return rules

    # Hasilkan aturan asosiasi
    # min_threshold untuk lift biasanya 1, yang menunjukkan hubungan positif
    try:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        logging.info(f"Jumlah aturan asosiasi yang ditemukan: {len(rules)}")
    except Exception as e:
        logging.error(f"Error during association_rules generation: {e}")
        return rules

    # Urutkan dan tampilkan 10 rekomendasi teratas
    return rules.sort_values(by='lift', ascending=False).head(10)


# ===== ROUTING =====
@app.route('/')
def index():
    df = load_data()
    if df.empty:
        return render_template('result.html',
                               visualizations={},
                               cluster_info={},
                               cluster_viz=None,
                               rules_table="<p class='warning-message'>Tidak ada data yang tersedia untuk analisis. Pastikan file 'Reviews.csv' ada dan tidak kosong.</p>",
                               transactions_for_debug="<p class='warning-message'>Tidak ada data transaksi untuk ditampilkan.</p>")

    visualizations = generate_visualizations(df)

    clustered_users_df, cluster_viz = cluster_users(df)

    cluster_summary = {}
    if not clustered_users_df.empty:
        cluster_summary = clustered_users_df['Cluster'].value_counts().sort_index().to_dict()
        logging.info(f"Ringkasan Cluster: {cluster_summary}")
    else:
        logging.warning("clustered_users_df kosong, tidak ada ringkasan cluster.")

    rules = generate_recommendations(df)

    rules_table = "<p class='warning-message'>Tidak ada aturan rekomendasi yang dihasilkan. Ini mungkin karena data terlalu jarang atau parameter min_support terlalu tinggi. Coba tingkatkan ukuran dataset atau turunkan min_support.</p>"
    if not rules.empty:
        # Konversi frozenset ke string untuk tampilan HTML yang lebih baik
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_table = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_html(
            classes='table table-striped table-bordered', index=False)
        logging.info("Generated Association Rules table.")

    # Siapkan data transaksi untuk debugging jika diperlukan
    transactions_for_debug = "<p class='warning-message'>Tidak ada data transaksi untuk ditampilkan (df kosong).</p>"
    if not df.empty:
        transactions_for_debug = df.groupby('UserId')['ProductId'].apply(lambda x: ', '.join(x)).reset_index().head(
            20).to_html(classes='table table-striped table-bordered', index=False)
        logging.info("Generated Transactions for Debugging table.")

    return render_template('result.html',
                           visualizations=visualizations,
                           cluster_info=cluster_summary,
                           cluster_viz=cluster_viz,
                           rules_table=rules_table,
                           transactions_for_debug=transactions_for_debug)


if __name__ == '__main__':
    # Pastikan Anda memiliki direktori 'dataset' dan file 'Reviews.csv' di dalamnya
    # Jalankan aplikasi Flask
    app.run(debug=True)