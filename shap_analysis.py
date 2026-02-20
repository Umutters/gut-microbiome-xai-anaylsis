import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Veriyi Oku (Dosya yolunu kontrol et)
try:
    df = pd.read_csv('Wu2020_Simulated_Data.csv')
except FileNotFoundError:
    print("Hata: CSV dosyası bulunamadı. Lütfen dosya adını kontrol edin.")
    # Test için rastgele veri oluşturulması gerekirse burayı açabilirsin:
    # df = pd.DataFrame(np.random.rand(100, 5), columns=['F1','F2','F3','F4','Group'])
    # df['Group'] = np.random.choice(['Healthy', 'T2D'], 100)
    exit()

# 2. Veriyi Hazırla
le = LabelEncoder()
df['Group_Encoded'] = le.fit_transform(df['Group'])

X = df.drop(columns=['Group', 'Group_Encoded'])
y = df['Group_Encoded']

# 3. Modeli Eğit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. SHAP Analizi
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# --- KRİTİK DÜZELTME BURADA ---
# T2D sınıfının indeksini buluyoruz
try:
    t2d_index = list(le.classes_).index('T2D')
except ValueError:
    print("T2D sınıfı bulunamadı, mevcut sınıflar:", list(le.classes_))
    t2d_index = 1 # Varsayılan olarak 1. indexi al (Genelde pozitif sınıf)

# Explanation nesnesini dilimliyoruz: [Bütün Satırlar, Bütün Özellikler, T2D İndeksi]
# Bu işlem bize sadece T2D sınıfı için olan SHAP değerlerini verir.
shap_values_t2d = shap_values[:, :, t2d_index]

# Grafik 1: Bar Plot (Ortalama mutlak etki)
plt.figure()
shap.summary_plot(shap_values_t2d, X_test, plot_type="bar", show=False)
plt.title(f"SHAP Özellik Önemi (Sınıf: {le.classes_[t2d_index]})")
plt.show()

# Grafik 2: Beeswarm Plot (Detaylı etki yönü)
plt.figure()
shap.summary_plot(shap_values_t2d, X_test, show=False)
plt.title(f"SHAP Detay Grafiği (Sınıf: {le.classes_[t2d_index]})")
plt.show()