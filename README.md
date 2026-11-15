# Yoklama Sistemi

C# ve Windows Forms ile yazılmış, kamera üzerinden yüz tanıyan bir yoklama uygulaması.
Yüz bulma için OpenCV Haar Cascade, tanıma için LBPH algoritması ([Emgu CV](https://www.emgu.com/) üzerinden) kullanılır.

## Gereksinimler

- Windows 10/11 (64 bit)
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- Bir web kamerası

## Çalıştırma

```
cd WindowsFormsApp56
dotnet run
```

İlk çalıştırmada gerekli NuGet paketleri (Emgu CV) otomatik indirilir.

## Kullanım

1. Kişinin adını yaz.
2. **Kamerayı Başlat**'a bas.
3. Kameraya bak ve başını hafifçe çevirerek **Yüzü Kaydet**'e 10-20 kez bas.
4. **Eğit**'e bas. Tanıma otomatik açılır ve yüzün üstünde isim görünür.
5. Diğer kişiler için aynı adımları tekrarla, ardından yeniden **Eğit**'e bas.

Kaydedilen yüzler, programın çalıştığı klasördeki `TrainedFaces` klasörüne `Isim_1.png`, `Isim_2.png` ... şeklinde kaydedilir. Program her açıldığında bu resimlerle tekrar eğitilir.

## Ayar

Tanıma hassasiyeti `Classifier_Train.cs` dosyasındaki `Threshold` değeriyle ayarlanır (varsayılan `80`):

- Yabancılar tanıdık biri olarak görünüyorsa değeri **düşür**.
- Tanıdık kişiler "Bilinmiyor" çıkıyorsa değeri **yükselt**.

## Proje yapısı

| Dosya | Görevi |
|---|---|
| `Form1.cs` | Arayüz, kamera ve butonlar |
| `BusinessRecognition.cs` | Karede yüz bulma ve isim çizme |
| `Classifier_Train.cs` | Yüz kaydetme, LBPH eğitimi ve tanıma |
| `haarcascade_frontalface_default.xml` | OpenCV yüz bulma modeli |
