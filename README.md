Setup:

Skimage: conda install -c anaconda scikit-image
Sklearn : pip install -U scikit-learn
Keras : conda install -c conda-forge keras
tensorflow : conda install -c conda-forge tensorflow
Bu kütüphanelerden herhangi biri yüklü değilse anaconda command promptan yukarıdakiler yazılarak yüklenebilir, yüklenmezse
kütüphanelerin kendi sitelerinde alternatif olarak yazılması gerekenler var.


Kodları çalıştırırken dikkat edilmesi gerekenler: 

num class : kaç sınıflı çalışma yapılacaksa o belirlenir.

directory : kod dosyasıyla aynı directoryde dataset adında dosya olması ve o dosyanın içinde resimler her sınıfın
kendine ait klasörlerinde yer alması gerekmekte.

resimlerin formatına göre 2. for döngüsünün içindeki if file_name.endswith(".tif"): kısmında .tif değil örneğin .jpeg , .png
diye değiştirilmeli

