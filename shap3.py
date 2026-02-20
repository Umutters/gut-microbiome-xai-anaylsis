# Gerekli kütüphaneleri yükle (Eğer yüklü değilse)
# !pip install shap pandas numpy matplotlib scikit-learn

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. DOSYALARI YÜKLE
# Dosya isimlerinin senin kaydettiklerinle aynı olduğundan emin ol
print("Veriler yükleniyor...")
abundance = pd.read_csv("KarlssonFH_2013_Abundance.csv", index_col=0)
metadata = pd.read_csv("KarlssonFH_2013_Metadata.csv", index_col=0)

# 2. VERİYİ HAZIRLA (Transpose ve Temizlik)
# Veriyi çevir: Satırlar Hasta, Sütunlar Bakteri olsun
X = abundance.T

# Bakteri isimlerini sadeleştir (sadece tür ismini al)
# Örnek: "k__Bacteria|...|s__Roseburia_intestinalis" -> "Roseburia_intestinalis"
X.columns = [col.split('|')[-1].replace("s__", "") for col in X.columns]

# Metadata ile birleştir (index/hasta ID'leri üzerinden)
# Metadata dosyasında hastalık bilgisinin 'disease' sütununda olduğunu varsayıyoruz
merged = X.merge(metadata[['disease']], left_index=True, right_index=True)

# 3. SADECE İLGİLİ GRUPLARI FİLTRELE
# Çalışmamız için gerekli gruplar: IGT (Prediyabet), T2D (Diyabet), NGT (Sağlıklı)
target_groups = ['IGT', 'T2D', 'NGT']
filtered = merged[merged['disease'].isin(target_groups)].copy()

print(f"Analiz edilen hasta sayısı: {filtered.shape[0]}")
print(f"İncelenen bakteri sayısı: {X.shape[1]}")

# Özellikler (X) ve Hedef (y) ayrımı
X_final = filtered.drop(columns=['disease'])
y = filtered['disease']

# Hedef değişkeni sayısal hale getir (0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Sınıf isimlerini kontrol et
print("\nSınıf Etiketleri:")
for i, name in enumerate(le.classes_):
    print(f"{i}: {name}")

# 4. MODELİ EĞİT (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Model Başarısı
acc = accuracy_score(y_test, model.predict(X_test))
print(f"\nModel Doğruluğu: %{acc * 100:.2f}")

# 5. SHAP ANALİZİ (IGT - PREDİYABET ODAKLI)
print("\nSHAP analizi yapılıyor...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Prediyabet (IGT) sınıfının indeksini bul
try:
    igt_index = list(le.classes_).index('IGT')
    target_class_name = "IGT (Prediyabet)"
except ValueError:
    print("Uyarı: IGT sınıfı bulunamadı, T2D analiz ediliyor.")
    igt_index = list(le.classes_).index('T2D')
    target_class_name = "T2D (Tip 2 Diyabet)"

# Çok sınıflı çıktıdan sadece hedef sınıfın (IGT) değerlerini al
# SHAP versiyonuna göre yapı farklı olabilir, her ihtimali kapsayalım:
if isinstance(shap_values, list):
    shap_vals_target = shap_values[igt_index]
elif len(shap_values.shape) == 3:
    shap_vals_target = shap_values[:, :, igt_index]
else:
    shap_vals_target = shap_values

# 6. GRAFİĞİ ÇİZ VE KAYDET
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_vals_target, X_test, show=False)
plt.title(f"Karlsson (2013) Verisi: {target_class_name} Riskini Artıran/Azaltan Bakteriler", fontsize=16)
plt.tight_layout()
plt.show()