# 🩺 Çoklu İlaç Yan Etki Yapay Zekası: 2 Seviyeli Model Rehberi

Bu döküman, **aynı anda birden fazla ilaç kullanan hastalarda ortaya çıkabilecek riskleri** tahmin eden yapay zeka modelimizin **2 Seviyeli (Hiyerarşik)** karar mekanizmasını özetlemektedir.

---

## 🎯 Neden 2 Seviyeli Bir Mimari Kuruldu?
Tıpta binlerce farklı yan etki vardır. Bir doktora doğrudan yüzlerce küçük yan etki listesi vermek kafa karışıklığına ve **"alarm yorgunluğuna"** yol açar. 
Modelimiz tıpkı bir **uzman hekim gibi** iki aşamada düşünür:
1. **Önce büyük resme bakar:** *"Hangi organ tehlikede?"* (Makro Risk)
2. **Sonra detaya iner:** *"O organda tam olarak hangi hastalık/yan etki oluşacak?"* (Mikro Teşhis)

---

```mermaid
flowchart TD
    subgraph Input[1. İLAÇ GİRDİSİ]
        D["💊 İlaç A + 💊 İlaç B"]
    end
    
    subgraph L1[🏛️ SEVİYE 1: ORGAN SİSTEMİ RİSKİ (MAKRO)]
        O["%89.26 Doğruluk | %96.02 AUROC\n15 Ana Organ ve Sistem Değerlendirilir\n(Kalp, Karaciğer, Böbrek, Mide, Beyin vb.)"]
    end
    
    subgraph L2[🔬 SEVİYE 2: SPESİFİK YAN ETKİ TEŞHİSİ (MİKRO)]
        S["%84.82 Doğruluk | %88.52 AUROC\n100 Spesifik Klinik Tanı Detaylandırılır\n(Aritmi, Sarılık, Mide Kanaması, Titreme vb.)"]
    end
    
    Input --> L1
    L1 --> L2
```

---

## 🏛️ SEVİYE 1: 15 MedDRA Organ Sistemi Risk Değerlendirmesi (Makro Seviye)

* **Başarım Oranı:** **`%89.26 Sınıflandırma Doğruluğu`** | **`%96.02 Ayırt Edicilik (AUROC)`**
* **Ne Yapar?** İki ilaç vücuda girdiğinde uluslararası tıp standardı olan **15 ana organ ve vücut sisteminin** hangisinde toksisite veya hasar riski olduğunu tespit eder.
* **İncelenen 15 Temel Sistem:**
  1. **Kalp ve Damar:** Kalp krizi, ritim bozukluğu, tansiyon riskleri.
  2. **Karaciğer (Hepatobiliyer):** Karaciğer enzim yükü ve doku toksisitesi.
  3. **Mide ve Bağırsak (Gastrointestinal):** Sindirim sistemi, ülser ve kanamalar.
  4. **Böbrek ve Boşaltım (Renal):** Böbrek yetmezliği ve süzme bozukluğu.
  5. **Sinir Sistemi ve Beyin:** Felç, bilinç kaybı, nöropati.
  6. **Solunum (Akciğerler):** Nefes darlığı, astım tetiklenmesi.
  7. **Kan ve Lenf:** Anemi, pıhtılaşma bozuklukları, kanama.
  8. **Deri ve Cilt:** Ağır alerjik döküntüler, lezyonlar.
  9. **Psikiyatrik:** Depresyon, anksiyete, uyku bozuklukları.
  10. **Metabolizma & Hormonlar:** Şeker koması, tiroid, tuz-sıvı dengesizliği.
  11. **Kas-İskelet:** Kas erimesi (rabdomiyoliz), eklem hasarı.
  12. **Bağışıklık Sistemi:** Şok, anafilaksi, otoimmün reaksiyonlar.
  13. **Göz ve Kulak:** Görme kaybı, kulak çınlaması.
  14. **Enfeksiyon:** Bağışıklık baskılanması sonucu enfeksiyon yatkınlığı.
  15. **Genel Vücut:** Yüksek ateş, kronik halsizlik, ödem.

---

## 🔬 SEVİYE 2: 100 Spesifik Yan Etki Teşhisi (Mikro Seviye)

* **Başarım Oranı:** **`%84.82 Sınıflandırma Doğruluğu`** | **`%88.52 Ayırt Edicilik (AUROC)`**
* **Ne Yapar?** Tehlike tespit edilen organ altında oluşabilecek **en kritik 100 spesifik teşhisi** tek tek puanlar ve olasılıklarını sıralar.

---

## 📋 Örnek Klinik Senaryo (Doktorun Ekranında Nasıl Görünür?)

Bir hasta **İlaç A** ve **İlaç B** reçete edildiğinde yapay zekanın ürettiği rapor:

> ### ⚠️ POLİFARMASİ RİSK RAPORU
> 
> **1. ADIM (Organ Seviyesi Uyarısı - %89.26 Güvenilirlik):**
> * 🔴 **Karaciğer (Hepatobiliyer):** **%94 YÜKSEK RİSK**
> * 🔴 **Kalp-Damar Sistemi:** **%88 YÜKSEK RİSK**
> * 🟢 **Böbrekler:** %12 Güvenli
> * 🟢 **Solunum:** %5 Güvenli
> 
> **2. ADIM (Spesifik Yan Etki Detayları):**
> * *Karaciğer Hasarı Altında:* **ALT/AST Enzim Artışı (%92)**, **Sarılık (%85)**
> * *Kalp-Damar Altında:* **Taşikardi (%89)**, **Hipotansiyon (%81)**

---

## 💡 Bu Modelin Sağladığı 3 Büyük Fayda

1. **Hayat Kurtarır:** Birbiriyle ölümcül etkileşime giren ilaçları saniyeler içinde yakalar.
2. **Alarm Yorgunluğunu Önler:** Doktoru binlerce alakasız maddeye boğmaz; doğrudan etkilenen organa odaklar.
3. **Bilimsel Güç:** Dünyanın en büyük biyomedikal grafı (Harvard PrimeKG) ve 2.4 milyon molekül (ChEMBL) üzerinde eğitilmiştir.
