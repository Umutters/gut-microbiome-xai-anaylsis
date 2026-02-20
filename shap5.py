import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# 1. VERİLERİ OKU
print("Dosyalar okunuyor...")
abundance = pd.read_csv("KarlssonFH_2013_Abundance.csv", index_col=0)
metadata = pd.read_csv("KarlssonFH_2013_Metadata.csv", index_col=0)

# 2. İSİM KONTROLÜ VE TEMİZLİK (Kritik Adım)
X = abundance.T
# Bakteri isimlerini temizle
X.columns = [col.split('|')[-1].replace("s__", "") for col in X.columns]

# İndekslerdeki boşlukları temizle (Bazen "S112 " gibi boşluklar merge'i bozar)
X.index = X.index.str.strip()
metadata.index = metadata.index.str.strip()

print(f"Bakteri Tablosu Örnek Sayısı: {X.shape[0]}")
print(f"Metadata Tablosu Örnek Sayısı: {metadata.shape[0]}")

# 3. GÜVENLİ BİRLEŞTİRME (MERGE)
# 'inner' join yerine kontrol için şimdilik 'left' veya direkt index eşleşmesi yapalım
merged = X.merge(metadata[['disease']], left_index=True, right_index=True, how='inner')

print(f"Birleştirme Sonrası Örnek Sayısı: {merged.shape[0]}")

if merged.empty:
    print("!!! HATA: Birleştirme sonucu boş! İndeks isimleri uyuşmuyor.")
    print("Bakteri İndeks Örneği:", X.index[0:5].tolist())
    print("Metadata İndeks Örneği:", metadata.index[0:5].tolist())
    exit()

# 4. GRUP FİLTRELEME
# Hedef gruplar: NGT (Sağlıklı) ve IGT (Prediyabet)
target_pair = ['NGT', 'IGT']
binary_data = merged[merged['disease'].isin(target_pair)].copy()

print("\n--- Sınıf Dağılımı ---")
print(binary_data['disease'].value_counts())

# Eğer IGT (Prediyabet) yoksa işlemi durdur
if 'IGT' not in binary_data['disease'].unique():
    print("!!! KRİTİK HATA: Veri setinde 'IGT' (Prediyabet) hastası bulunamadı!")
    print("Mevcut etiketler:", binary_data['disease'].unique())
    # IGT yoksa kodun geri kalanı çalışmaz, burada durmalı
else:
    # 5. MODEL HAZIRLIĞI
    X_bin = binary_data.drop(columns=['disease'])
    y_bin = binary_data['disease']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_bin)
    
    # Hangi sınıfın 1 olduğunu bulalım
    if len(le.classes_) > 1:
        positive_class = le.inverse_transform([1])[0]
        print(f"\nModel Hedefi (1): {positive_class}")
    else:
        print("Hata: Sadece tek bir sınıf var, model eğitilemez.")
        exit()

    # 6. MODEL EĞİTİMİ (Logistic Regression - L1 Regularization)
    # L1 (Lasso) kullandım çünkü gereksiz bakterileri eler, doğruluk artar.
    model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
    
    # Cross Validation Skoru
    scores = cross_val_score(model, X_bin, y_encoded, cv=5, scoring='accuracy')
    print(f"\nOrtalama Doğruluk (Accuracy): %{scores.mean()*100:.2f}")

    # Tüm veriyle eğit ve SHAP çiz
    model.fit(X_bin, y_encoded)
    
    print("\nSHAP grafiği çiziliyor...")
    explainer = shap.LinearExplainer(model, X_bin)
    shap_values = explainer.shap_values(X_bin)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_bin, show=False)
    plt.title(f"{positive_class} (Prediyabet) ile İlişkili Bakteriler", fontsize=16)
    plt.show()