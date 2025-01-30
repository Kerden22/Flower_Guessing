
# Çiçek Sınıflandırma Projesi (CNN ve Transfer Learning)

## 🌸 **Proje Genel Bakışı**

Bu proje, çiçek görsellerini beş farklı kategoriye sınıflandırmayı amaçlar ve iki farklı derin öğrenme yöntemini kullanır:

1.  **Özel bir Konvolüyonel Sinir Ağı (CNN)**.
2.  **Önceden eğitilmiş ResNet-18 ile Transfer Learning**.

Bu proje, **veri yükleme ve önişlemeden** başlayıp, **modellerin eğitilmesi**, **performansının değerlendirilmesi** ve **yeni görüntülerde tahmin yapmaya** kadar tüm aşamaları kapsar.

## 🌼 **Veri Seti**

Veri seti, çiçek görsellerinden oluşur ve beş sınıfa ayrılmıştır:

-   Papatya (Daisy)
-   Karahindiba (Dandelion)
-   Gül (Rose)
-   Ayçiçeği (Sunflower)
-   Lale (Tulip)



## 🔧 **Gereksinimler**

Projeyi çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyacınız var:

```
torch
torchvision
matplotlib
scikit-learn
Pillow

```

Bunları aşağıdaki komutla yükleyebilirsiniz:

```
pip install torch torchvision matplotlib scikit-learn pillow
```

----------

## 📊 **Model Mimarisinin Detayları**

### **Özel CNN Modeli**

Bu CNN modeli şunları içerir:

-   **2 konvolüyonel katman** ReLU aktivasyon ve max pooling ile. 
-   **2 fully connected katman**, özellikleri sınıf tahminine dönüştürmek için. (Katman sayıları artırılabilir.)

### **ResNet-18 Transfer Learning**

-   **ImageNet** üzerinde önceden eğitilmiş olup, son fully connected katman 5 çiçek sınıfına uyarlanır.
-   Özellikle özel CNN’e göre daha hızlı ve genelde daha iyi sonuçlar verir.

----------

## 🏋️ **Eğitim Süreci**

-   **Epoch:** 5 (değiştirilebilir)
-   **Batch Boyutu:** 32
-   **Optimizasyon:** Adam, öğrenme oranı 0.001
-   **Kâyıp Fonksiyonu:** Cross-Entropy Loss

Her iki model de **%80 eğitim, %20 test** olarak bölünmüş veri ile eğitildi.

----------

## 📊 **Değerlendirme Metrikleri**

Modeller şunlara göre değerlendirilmiştir:

-   **Doğruluk (Accuracy)**
-   **Kesinlik (Precision)**
-   **Duyarlılık (Recall)**
-   **F1-skoru**

Ayrıca, **scikit-learn** kullanılarak sınıf bazlı değerlendirme raporları oluşturulmuştur.


----------

## 🔍 **Yeni Görüntülerde Tahmin**

Bir çiçek görselinin sınıfını tahmin etmek için:

```python
from PIL import Image
from torchvision import transforms
import torch

# Modeli ve görüntüyü yükle
image_path = "gorsel_yolu.jpg"
image, predicted_class = predict_image(model, image_path, class_names)

print(f"Tahmin Edilen Sınıf: {predicted_class}")

```

Bu kod, tahmin edilen sınıfı yazdırır ve görseli Matplotlib ile görüntüler.



----------

## 🛠️ **Potansiyel İyileştirmeler**

-   **Veri Çoğaltma (Augmentasyon):** Döndürme, aynalama gibi tekniklerle veri seti zenginleştirilebilir.
-   **Hiperparametre Optimizasyonu:** Öğrenme oranı, batch boyutu ve optimizasyon algoritmaları test edilebilir.
-   **Ek Katmanlar:** Daha fazla konvolüyonel katman eklenebilir veya farklı mimariler denenebilir.

----------


-   [Mahmut Kerem Erden]

Projeyi fork edebilir, pull request oluşturabilir veya öneriler için issue açabilirsiniz!
