buat virtual env terlebih dahulu dengan perintah `python -m venv venv`.
aktifkan virtual env dengan perintah `source venv/bin/activate` (Linux) atau `call .\venv\Scripts\activate.bat` (Windows).
setelah aktif virtual env, install library yang dibutuhkan.
```
pip install nltk
pip install gradio
pip install tensorflow
pip install Sastrawi
pip install sentence-transformers
pip install transformers
```
setelah install library, aktifkan server dengan perintah `python main.py`.