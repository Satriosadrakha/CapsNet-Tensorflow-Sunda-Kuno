# CapsNet-Tensorflow

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=plastic)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](https://opensource.org/licenses/Apache-2.0)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg?style=plastic)](https://gitter.im/CapsNet-Tensorflow/Lobby)

A Tensorflow implementation of CapsNet based on Geoffrey Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

![capsVSneuron](imgs/capsuleVSneuron.png)

> **Notes:**
> 1. The current version supports [MNIST](http://yann.lecun.com/exdb/mnist/) and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) datasets. The current test accuracy for MNIST is `99.64%`, and Fashion-MNIST `90.60%`, see details in the [Results](https://github.com/naturomics/CapsNet-Tensorflow#results) section
> 2. See [dist_version](dist_version) for multi-GPU support
> 3. [Here(知乎)](https://zhihu.com/question/67287444/answer/251460831) is an article explaining my understanding of the paper. It may be helpful in understanding the code.


> **Important:**
>
> If you need to apply CapsNet model to your own datasets or build up a new model with the basic block of CapsNet, please follow my new project [CapsLayer](https://github.com/naturomics/CapsLayer), which is an advanced library for capsule theory, aiming to integrate capsule-relevant technologies, provide relevant analysis tools, develop related application examples, and promote the development of capsule theory. For example, you can use capsule layer block in your code easily with the API ``capsLayer.layers.fully_connected`` and ``capsLayer.layers.conv2d``


## Requirements
- Python
- NumPy
- [Tensorflow](https://github.com/tensorflow/tensorflow)>=1.3
- tqdm (for displaying training progress info)
- scipy (for saving images)

## Usage
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

### Reference
- [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)