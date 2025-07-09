import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import skfuzzy.control as ctrl


st.title('Prediksi Harga Jual, Estimasi Masa Simpan, dan Kualitas Buah Pisang 🍌')
st.write(' ')
st.sidebar.title('✨ Metode SPK : Fuzzy ✨')
st.sidebar.write('#### 😺 Assasa Salma')
st.sidebar.write('#### 🐹 Mufidah Shofi Aqila')
tampilan = st.sidebar.selectbox('Pilih Menu', ["Lihat Tampilan Tabel Dataset", "Hitung Prediksi"])

# import dataset yang dibutuhkan
data = pd.read_csv('banana_quality.csv', nrows=5000)
df = pd.DataFrame(data)

# variabel fuzzy
## variabel input
size = ctrl.Antecedent(np.arange(float(df['Size'].min()), float(df['Size'].max()) + 0.1, 0.1),'Size')
weight = ctrl.Antecedent(np.arange(float(df['Weight'].min()), float(df['Weight'].max()) + 0.1, 0.1),'Weight')
sweetness = ctrl.Antecedent(np.arange(float(df['Sweetness'].min()), float(df['Sweetness'].max()) + 0.1, 0.1),'Sweetness')
softness = ctrl.Antecedent(np.arange(float(df['Softness'].min()), float(df['Softness'].max()) + 0.1, 0.1),'Softness')
harvestTime = ctrl.Antecedent(np.arange(float(df['HarvestTime'].min()), float(df['HarvestTime'].max()) + 0.1, 0.1),'HarvestTime')
ripeness = ctrl.Antecedent(np.arange(float(df['Ripeness'].min()), float(df['Ripeness'].max()) + 0.1, 0.1),'Ripeness')
acidity = ctrl.Antecedent(np.arange(float(df['Acidity'].min()), float(df['Acidity'].max()) + 0.1, 0.1),'Acidity')

## variabel output
quality = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'Quality')
price = ctrl.Consequent(np.arange(8000, 75000, 1000), 'Price')
shelfLife = ctrl.Consequent(np.arange(0, 31, 1), 'ShelfLife')

# fungsi kluster
def fuzzy_clustering(df, antecedent, column_name, n_clusters, mf_types, labels):
    # membuat klustering pake fuzzy c-means
    values = df[column_name].values.reshape(1, -1)

    # klastering dgn jumlah kluster yg dinamis
    centers, _, _, _, _, _, _ = fuzz.cluster.cmeans(
        values, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
    )

    # pusat klaster diurutin
    sc = sorted(centers.flatten())

    if len(labels) != n_clusters:
        raise ValueError(f'Jumlah label ({len(labels)}) harus sama dengan jumlah klaster : {n_clusters}!')

    # menentukan fungsi keanggotaan
    for i in range(n_clusters):
        mf_type = mf_types.get(labels[i], "trimf")

        if mf_type == "trimf":
            antecedent[labels[i]] = fuzz.trimf(
                antecedent.universe, [sc[i-1] if i > 0 else sc[i], sc[i], sc[i+1] if i < n_clusters - 1 else sc[i] + 0.1]
            )
        elif mf_type == "trapmf":
            antecedent[labels[i]] = fuzz.trapmf(
                antecedent.universe, [sc[i-1] - 0.1 if i > 0 else sc[i] - 0.1, sc[i], sc[i+1] if i < n_clusters - 1 else sc[i], sc[i+1] + 0.1 if i < n_clusters - 1 else sc[i] + 0.1]
            )
        elif mf_type == "gaussmf":
            sigma = (sc[i+1] - sc[i]) / 2 if i < n_clusters - 1 else (max(values.flatten()) - sc[i]) / 2
            antecedent[labels[i]] = fuzz.gaussmf(
                antecedent.universe, mean=sc[i], sigma=sigma
            )
    return sc

mf_types_size = {
    "kecil": "gaussmf",
    "sedang": "gaussmf",
    "besar": "gaussmf"
}
sorted_size = fuzzy_clustering(df, size, 'Size', n_clusters=3, mf_types=mf_types_size, labels=['kecil', 'sedang', 'besar'])

mf_types_weight = {
    "ringan": "trapmf",
    "cukup": "trapmf",
    "berat": "trapmf"
}
sorted_weight = fuzzy_clustering(df, weight, 'Weight', n_clusters=3, mf_types=mf_types_weight, labels=['ringan', 'cukup', 'berat'])

mf_types_sweetness = {
    "tidak manis": "gaussmf",
    "cukup manis": "gaussmf",
    "manis": "gaussmf"
}
sorted_sweetness = fuzzy_clustering(df, sweetness, 'Sweetness', n_clusters=3, mf_types=mf_types_sweetness, labels=['tidak manis', 'cukup manis', 'manis'])

mf_types_softness = {
    "keras": "gaussmf",
    "lembut": "gaussmf"
}
sorted_softness = fuzzy_clustering(df, softness, 'Softness', n_clusters=2, mf_types=mf_types_softness, labels=['keras', 'lembut'])

mf_types_harvestTime = {
    "baru dipanen": "trimf",
    "1-3 hari": "trimf",
    "lama": "trimf"
}
sorted_harvestTime = fuzzy_clustering(df, harvestTime, 'HarvestTime', n_clusters=3, mf_types=mf_types_harvestTime, labels=['baru dipanen', '1-3 hari', 'lama'])

mf_types_ripeness = {
    "mentah": "gaussmf",
    "matang": "gaussmf",
    "terlalu matang": "gaussmf"
}
sorted_ripeness = fuzzy_clustering(df, ripeness, 'Ripeness', n_clusters=3, mf_types=mf_types_ripeness, labels=['mentah', 'matang', 'terlalu matang'])

mf_types_acidity = {
    "rendah": "gaussmf",
    "sedang": "gaussmf",
    "tinggi": "gaussmf"
}
sorted_acidity = fuzzy_clustering(df, acidity, 'Acidity', n_clusters=3, mf_types=mf_types_acidity, labels=['rendah', 'sedang', 'tinggi'])


## var output
quality['rendah'] = fuzz.trimf(quality.universe, [0,0,5])
quality['cukup'] = fuzz.trapmf(quality.universe, [0,3,7,10])
quality['tinggi'] = fuzz.trimf(quality.universe, [5,10,10])

price['murah'] = fuzz.trimf(price.universe, [8000, 8000, 20000])
price['standar'] = fuzz.trimf(price.universe, [15000, 20000, 45000])
price['mahal'] = fuzz.trimf(price.universe, [40000, 75000, 75000])

shelfLife['pendek'] = fuzz.trapmf(shelfLife.universe, [0, 0, 2, 4])
shelfLife['sedang'] = fuzz.trapmf(shelfLife.universe, [0, 3, 5, 7])
shelfLife['panjang'] = fuzz.trapmf(shelfLife.universe, [6, 10, 30, 30])


rules = [
    ctrl.Rule(size['kecil'] & weight['ringan'] & sweetness['tidak manis'] & softness['keras'] & harvestTime['baru dipanen'] & ripeness['mentah'] & acidity['tinggi'], consequent=[quality['rendah'], price['murah'], shelfLife['panjang']]),
    ctrl.Rule(size['sedang'] & weight['cukup'] & sweetness['cukup manis'] & softness['lembut'] & harvestTime['1-3 hari'] & ripeness['matang'] & acidity['sedang'], consequent=[quality['cukup'], price['standar'], shelfLife['sedang']]),
    ctrl.Rule(size['besar'] & weight['berat'] & sweetness['manis'] & softness['lembut'] & harvestTime['lama'] & ripeness['terlalu matang'] & acidity['rendah'], consequent=[quality['tinggi'], price['mahal'], shelfLife['pendek']]),
    ctrl.Rule(size['sedang'] & weight['berat'] & sweetness['cukup manis'] & softness['keras'] & harvestTime['1-3 hari'] & ripeness['matang'] & acidity['sedang'], consequent=[quality['cukup'], price['standar'], shelfLife['sedang']]),
    ctrl.Rule(size['kecil'] & weight['ringan'] & sweetness['tidak manis'] & softness['keras'] & harvestTime['baru dipanen'] & ripeness['mentah'] & acidity['tinggi'], consequent=[quality['rendah'], price['murah'], shelfLife['panjang']]),
    ctrl.Rule(size['besar'] & weight['berat'] & sweetness['manis'] & softness['lembut'] & harvestTime['lama'] & ripeness['matang'] & acidity['rendah'], consequent=[quality['tinggi'], price['mahal'], shelfLife['pendek']]),
    ctrl.Rule(size['sedang'] & weight['cukup'] & sweetness['cukup manis'] & softness['keras'] & harvestTime['1-3 hari'] & ripeness['matang'] & acidity['sedang'], consequent=[quality['cukup'], price['standar'], shelfLife['sedang']]),
    ctrl.Rule(size['besar'] & weight['berat'] & sweetness['manis'] & softness['keras'] & harvestTime['1-3 hari'] & ripeness['matang'] & acidity['rendah'], consequent=[quality['tinggi'], price['mahal'], shelfLife['sedang']]),
    ctrl.Rule(size['kecil'] & weight['ringan'] & sweetness['tidak manis'] & softness['keras'] & harvestTime['baru dipanen'] & ripeness['matang'] & acidity['tinggi'], consequent=[quality['rendah'], price['murah'], shelfLife['sedang']]),
    ctrl.Rule(size['sedang'] & weight['ringan'] & sweetness['cukup manis'] & softness['lembut'] & harvestTime['1-3 hari'] & ripeness['matang'] & acidity['sedang'], consequent=[quality['cukup'], price['standar'], shelfLife['sedang']])
]

system_ctrl = ctrl.ControlSystem(rules)
bananas = ctrl.ControlSystemSimulation(system_ctrl)


# fungsi untuk tampilan grafik
def plot_display_output(variable, title, result):
    fig, ax = plt.subplots()
    for term_name in variable.terms:
        term_mf = variable[term_name].mf
        ax.plot(variable.universe, term_mf, label=term_name)
    ax.grid(True, alpha=0.35)
    ax.set_title(title, loc='center', fontsize=14)
    ax.set_xlabel(variable.label)
    ax.set_ylabel("Derajat Keanggotaan")
    ax.axvline(result, color='black', linestyle='--', label=f'Hasil : {result:.2f}')
    ax.legend()
    st.pyplot(fig)


if tampilan == 'Lihat Tampilan Tabel Dataset':
    st.write('#### 📂 Dataset Kualitas Pisang')
    jmlh_dataset = st.slider('Jumlah data yang ingin ditampilkan dari dataset:', 0, len(df), 30)
    st.write(df.head(jmlh_dataset))
    # menampilkan seluruh dataset
    with st.expander('Detail Dataset') :
        st.write('Dimensi Dataset', df.shape)

        st.write('Dimensi Statistik : ')
        st.write(df.describe())

        st.write('Missing Value : ')
        st.write(df.isnull().sum())

    st.write('###')
    st.write('#### 📊 Visualisasi Data')

    # Visualisasi semua data
    # Ambil hanya kolom numerik
    numeric_cols = df.select_dtypes(include='number').columns
    # Plot semua kolom numerik jadi satu grafik
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in numeric_cols:
        ax.plot(df.index, df[col], label=col)
    ax.set_xlabel('Index')
    ax.set_ylabel('Nilai')
    ax.set_title('Line Plot Semua Kriteria')
    ax.legend()
    st.pyplot(fig)

    # Visualisasi per kriteria
    criteria = ['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity'] 
    colors =  ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'pink']
    # Plot satu per satu
    for col, color in zip(criteria, colors):
        st.write(f'#### Grafik Kriteria {col}')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df.index, df[col], color=color, label=col, alpha=0.6, edgecolor='k', linewidth=0.3)
        ax.set_xlabel('Index')
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)





    
    

elif tampilan == 'Hitung Prediksi':
    st.write('#### Input Nilai')

    # Input Nilai
    col1, col2 = st.columns(2)
    with col1:
        size_value = st.selectbox("📏 Ukuran (size)", ["Kecil", "Sedang", "Besar"], 0)
        sweetness_value = st.selectbox("🍰 Kemanisan (sweetness)", ["Tidak Manis", "Cukup Manis", "Manis"], 0)
        softness_value = st.selectbox("🥰 Kelembutan (softness)", ["Keras", "Lembut"], 0)
        harvestTime_value = st.selectbox("🌾 Lama setelah panen (harvestTime)", ["Baru Dipanen", "1-3 Hari", "Lama"], 0)

    with col2:
        weight_value = st.selectbox("⚖️ Berat (weight)", ["Ringan", "Cukup", "Berat"], 0)
        ripeness_value = st.selectbox(" 🍌Kematangan (ripeness)", ["Mentah", "Matang", "Terlalu Matang"], 0)
        acidity_value = st.selectbox("🍋‍🟩 Keasaman (acidity)", ["Rendah", "Sedang", "Tinggi"], 2)

    # Mengubah nilai pilihan menjadi nilai numerik dari nilai klustering + masukin ke input
    size_value = size_value.lower()
    size_mapping = dict(zip(["kecil", "sedang", "besar"], sorted_size))
    bananas.input['Size'] = size_mapping[size_value]

    sweetness_value = sweetness_value.lower()
    sweetness_mapping = dict(zip(["tidak manis", "cukup manis", "manis"], sorted_sweetness))
    bananas.input['Sweetness'] = sweetness_mapping[sweetness_value]

    softness_value = softness_value.lower()
    softness_mapping = dict(zip(["keras", "lembut"], sorted_softness))
    bananas.input['Softness'] = softness_mapping[softness_value]

    harvestTime_value = harvestTime_value.lower()
    harvestTime_mapping = dict(zip(["baru dipanen", "1-3 hari", "lama"], sorted_harvestTime))
    bananas.input['HarvestTime'] = harvestTime_mapping[harvestTime_value]

    weight_value = weight_value.lower()
    weight_mapping = dict(zip(["ringan", "cukup", "berat"], sorted_weight))
    bananas.input['Weight'] = weight_mapping[weight_value]

    ripeness_value = ripeness_value.lower()
    ripeness_mapping = dict(zip(["mentah", "matang", "terlalu matang"], sorted_ripeness))
    bananas.input['Ripeness'] = ripeness_mapping[ripeness_value]

    acidity_value = acidity_value.lower()
    acidity_mapping = dict(zip(["rendah", "sedang", "tinggi"], sorted_acidity))
    bananas.input['Acidity'] = acidity_mapping[acidity_value]

    bananas.compute()

    st.write('')
    if st.button('Hitung Prediksi'):
        st.write('####')
        st.write('### Nilai yang Diinputkan')
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Ukuran', 'Berat', 'Kemanisan', 'Kelembutan', 'Masa Panen', 'Kematangan', 'Keasaman'])
        with tab1:
            plot_display_output(size, 'Fungsi Keanggotaan Ukuran', size_mapping[size_value])
        with tab2:
            plot_display_output(weight, 'Fungsi Keanggotaan Berat', weight_mapping[weight_value])
        with tab3:
            plot_display_output(sweetness, 'Fungsi Keanggotaan Kemanisan', sweetness_mapping[sweetness_value])
        with tab4:
            plot_display_output(softness, 'Fungsi Keanggotaan Kelembutan', softness_mapping[softness_value])
        with tab5:
            plot_display_output(harvestTime, 'Fungsi Keanggotaan Masa Panen', harvestTime_mapping[harvestTime_value])
        with tab6:
            plot_display_output(ripeness, 'Fungsi Keanggotaan Kematangan', ripeness_mapping[ripeness_value])
        with tab7:
            plot_display_output(acidity, 'Fungsi Keanggotaan Keasaman', acidity_mapping[acidity_value])

        st.write('### Hasil Prediksi')
        tab1, tab2, tab3 = st.tabs(['Kualitas', 'Harga Jual', 'Masa Simpan'])
        with tab1:
            plot_display_output(quality, 'Fungsi Keanggotaan Kualitas', bananas.output['Quality'])
            st.success(f'### 📊 Hasil prediksi kualitas : {bananas.output['Quality']:.2f}')
        with tab2:
            plot_display_output(price, 'Fungsi Keanggotaan Harga Jual', bananas.output['Price'])
            st.success(f'### 🤑 Hasil prediksi harga jual : Rp{bananas.output['Price']:.0f} per sisir')
        with tab3:
            plot_display_output(shelfLife, 'Fungsi Keanggotaan Masa Simpan', bananas.output['ShelfLife'])
            st.success(f'### 🗄️Hasil prediksi masa simpan : {bananas.output['ShelfLife']:.1f} hari')



    





