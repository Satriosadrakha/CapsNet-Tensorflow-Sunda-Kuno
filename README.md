# CapsNet-Tensorflow-Sunda-Kuno

Implementasi CapsNet pada Tensorflow dengan aksara Sunda Kuno berdasarkan paper Geoffrey Hinton [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

## Persyaratan
- Python
- NumPy
- [Tensorflow](https://github.com/tensorflow/tensorflow)>=1.3
- tqdm
- scipy
- PIL

## Penggunaan
**Langkah 1.** Download repository ini dengan ``git``.

```
$ git clone https://github.com/Satriosadrakha/CapsNet-Tensorflow-Sunda-Kuno.git
$ cd CapsNet-Tensorflow-Sunda-Kuno
```

**Langkah 2.** Mulai training (Dataset sunda_kuno digunakan sebagai default):

```
$ python main.py
$ # atau training SleukRith-Set
$ python main.py --dataset khmer
```

**Langkah 3.** Kalkulasi akurasi test

```
$ python main.py --is_training=False --batch_size=1
$ # untuk SleukRith-Set
$ python main.py --dataset khmer --is_training=False --batch_size=1
```

> **Note:** Parameter default batch size adalah 24, dan epoch 40. File ``config.py`` perlu dimodifikasi atau gunakan parameter command line untuk menyesuaikannya, misal set batch size ke 64 and lakukan test summary setiap 200 step: ``python main.py  --test_sum_freq=200 --batch_size=48``

### Referensi
- [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)