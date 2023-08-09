# Capstone3_Telco_Customer_Churn
File pemodelan ini berisi tentang dataset Telco Customer Churn dengan jumlah 4930 baris dan 11 kolom. Pemodelan ini bertujuan untuk membuat model machine learning untuk memprediksi apakah suatu pelanggan akan berhenti berlangganan (churn) atau tidak. Pada file pemodelan ini tahapan-tahapan yang dilakukan adalah **_Business Understanding_**, _**Data Understanding**_, _**EDA**_, _**Data Preprocessing**_, _**Modeling**_, _**Model Properties**_ dan _**Conclusion & Recommendation**_.

# Business Understanding
***Context :***

Pada saat ini, industri telekomunikasi telah berkembang pesat. Banyak sekali perusahaan telekomunikasi yang berkompetisi menawarkan jasanya menggunakan sistem berlangganan untuk menjual jasanya, sehingga persaingan antar perusahaan semakin ketat. Salah satu tantangan yang kini dihadapi perusahaan adalah usaha untuk menurunkan jumlah pelanggan yang berhenti menggunakan layanan perusahaan dan beralih ke perusahaan kompetitor.

Suatu perusahaan ingin mengetahui pelanggan yang bagaimana yang akan pindah (*churn*) dari perusahaan tersebut sehingga jumlah pelanggan yang beralih (*churn*) dapat dikurangi. Seorang Data Scientist diminta untuk membuat model prediksi yang tepat untuk menentukan pelanggan akan berhenti menggunakan layanan (*churn*) atau tidak dengan menggunakan machine learning. 

Target :

0 : Tidak berhenti menggunakan layanan

1 : Berhenti menggunakan layanan (*churn*)

***Problem Statement :***

Pada suatu perusahaan telekomunikasi, tingginya persentase pelanggan yang melakukan *churn* menjadi salah satu indikator tingkat kegagalan suatu perusahaan telekomunikasi, sehingga diperlukan upaya untuk mengurangi persentase pelanggan *churn* tersebut. Pada umumnya perusahaan lebih memilih untuk mempertahankan pelanggan, karena biaya untuk mempertahankan pelanggan (*customer retention cost*) lebih rendah daripada memperoleh pelanggan baru (*customer acquisition cost*). Memperoleh pelanggan baru  dapat menghabiskan biaya **lima kali lebih besar** daripada mempertahakan pelanggan [(sumber 1)](https://www.optimove.com/resources/learning-center/customer-acquisition-vs-retention-costs) [(sumber 2)](https://www.linkedin.com/pulse/customer-retention-vs-acquisition-which-one-choose-myfundbox/). Sedangkan biaya untuk memperoleh pelanggan baru adalah sebesar $315 [(sumber 1)](https://salesworks.asia/media-centre/blog/customer-acquisition-cost-in-southeast-asia-whats-a-good-benchmark/) [(sumber 2)](https://startuptalky.com/cac-by-industry/).

Salah satu cara perusahaan telekomunikasi mempertahankan pelanggannya agar tidak *churn*, yaitu dengan memberikan insentif retensi terhadap pelanggan. Insentif retensi yang dimaksud adalah dengan memberikan suku bunga yang menarik, memberikan paket layanan yang menarik, memberikan prioritas pelayanan dan lain-lain dalam upaya untuk mempertahankan pelanggan. Namun, kebijakan pemberian insentif retensi belum sepenuhnya dilakukan secara efektif. Karena jika insentif retensi tersebut diberikan secara merata kepada seluruh pelanggan, maka pengeluaran biaya tersebut menjadi tidak efektif dan mengurangi potensi keuntungan apabila pelanggan tersebut memang loyal dan tidak ingin berhenti menggunakan layanan.

***Goals :***

Berdasarkan permasalahan di atas, perusahaan ingin memiliki kemampuan untuk memprediksi kemungkinan seorang pelanggan akan berhenti menggunakan layanan (*churn*) atau tidak, sehingga perusahaan dapat memfokuskan upaya retensi(mempertahankan pelanggan) pada pelanggan yang terindikasi untuk *churn*.
Selain itu, perusahaan ingin mengetahui faktor yang cenderung memperngaruhi pelanggan bertahan, sehingga mereka dapat membuat program bisnis yang tepat sasaran untuk mengurangi jumlah nasabah yang berhenti berlangganan (*churn*).

***Analytic Approach :***

Jadi yang akan dilakukan adalah menganalisis data untuk menemukan pola yang membedakan pelanggan yang akan berhenti menggunakan layanan (*churn*) atau tidak.

Kemudian akan membangun model klasifikasi yang akan membantu perusahaan untuk dapat memprediksi probabilitas seorang nasabah akan berhenti menggunakan layanan (*churn*) atau tidak.

***Metric Evaluation :***

Fokus utama dalam masalah ini pelanggan yang akan berhenti menggunakan layanan, maka target yang ditetapkan yaitu :
Target :
- 0 : Tidak berhenti menggunakan layanan
- 1 : Berhenti menggunakan layanan (*churn*)

Type 1 error : *False Positive* (pelanggan yang aktualnya tidak *churn* tetapi diprediksi *churn*)
Konsekuensi : tidak efektif dalam pemberian insentif

Type 2 error : *False Negative* (pelanggan yang aktualnya *churn* tetapi diprediksi tidak tidak akan *churn*)
Konsekuensi : kehilangan pelanggan

Berdasarkan konsekuensi yang ada, akan diberikan gambaran konsekuensi secara kuantitatif, maka akan dicoba perhitungan untuk mengatahui dampak biaya berdasarkan asumsi berikut :
- Tidak efektif dalam pemberian insentif retensi --> menyia-nyiakan biaya CRC (sebesar 0.2 kali biaya CAC) yaitu sebesar $63.
- Kehilangan pelanggan --> mengeluarkan biaya CAC (sebesar lima kali biaya CRC) yaitu sebesar $315.

Berdasarkan informasi yang ada, maka yang akan dilakukan adalah membuat model yang dapat mengurangi pelanggan *churn* dari perusahaan tersebut, khususnya *False Negative* (pelanggan yang aktualnya *churn* tetapi diprediksi tidak akan *churn*), tetapi juga dapat meminimalisir pemberian insentif retensi yang tidak tepat. Jadi metric utama yang akan digunakan adalah **f2_score**, karena pada kasus ini precision dianggap penting dan recall dianggap dua kali lebih penting daripada precision.

# ***Data Understanding***
Pada bab ini membahas tentang dataset yang digunakan, hal ini bertujuan untuk memahami data yang ada.

# **EDA**
Pada bab ini membahas tentang distribusi data pada dataset, hal ini bertujuan untuk menganalisis data berdasarkan keseluruhan nilai data pada setiap fiturnya. Pada bab ini juga dilakukan analisis untuk membandingkan pelanggan yang berhenti berlangganan atau tidak dengan tiap fitur pada dataset.

# ***Data Preprocessing***
Pada bab ini bertujuan untuk mempersiapkan data yang selanjutnya akan dilakukan pemodelan. Mempersiapkan yang dimaksud adalah menghapus duplikat data, mengecek apakah terdapat outlier, mempersiapkan fitur yang akan dilakukan _encoding_, _scaling_ yang dibutuhkan dalam pemodelan, mengecek apakah data target seimbang atau tidak dan hal apa yang akan dilakukan apabila data target tidak seimbang, dan mengubah nilai target churn = 1 dan tidak churn = 0.

# _**Modeling**_
Pada bab ini akan dilakukan pemodelan yang akan menggunakan beberapa algoritma klasifikasi yang ada, seperti Logistic Regression, KNeighbors, DecisionTree, RandomForest, AdaBoost, GradientBoosting dan XGB. Lalu memilih algoritma mana yang terbaik berdasarkan hasil validasi, test dan kestabilan tiap model.

# _**Model Properties**_
Pada bab ini akan membahas mengenai model mana yang menjadi final model pada pemodelan dataset ini. Pada bab ini juga akan menjelaskan tentang fitur-fitur yang penting menurut model, confusion matrix dan asumsi perhitung biaya apabila menggunakan model ini.

# _**Conclusion & Recommendation**_
Pada bab ini akan memaparkan kesimpulan dan rekomendasi berdasarkan hasil analisis dan pemodelan. Tidak lupa juga pada bab ini akan dijelaskan batasan-batasan yang ada apabila menggunakan model ini.
