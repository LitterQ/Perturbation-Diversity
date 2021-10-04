# PERTURBATION DIVERSITY CERTIFICATES ROBUST GENERALIZATION

## Introduction
This is the implementation of the
["PERTURBATION DIVERSITY CERTIFICATES ROBUST GENERALIZATION"].

The codes are implemented based on the released codes from ["Feature-Scattering Adversarial Training"](https://papers.nips.cc/paper/8459-defense-against-adversarial-attacks-using-feature-scattering-based-adversarial-training.pdf)


Tested under Python 3.7.9 and PyTorch 1.8.0.

### Train
Enter the folder named by the dataset you want to train. Specify the path for saving the trained models in ```fs_train.sh```, and then run
```
sh ./fs_train_cfiar10.sh  # for CIFAR-10 dataset
sh ./fs_train_svhn.sh     # for SVHN dataset
sh ./fs_train_cfiar100.sh # for CIFAR-100 dataset

```

### Evaluate
Specify the path to the trained models to be evaluated in ```fs_eval.sh``` and then run, using CIFAR-10 as a example. 
``` param: --init_model_pass: The load number of checkpoint, Possible values: `latest` for the last checkpoint, `199` for checkpoint-199 ``` 
```param: --attack_method_list: The attack list for evaluation, Possible values: `natural` for natural data, `fgsm`, `pgd`, `cw` ```
```
sh ./fs_eval_cifar10.sh


```

### Reference Model
A reference AT+PD model trained on [CIFAR-10](https://drive.google.com/file/d/1HyP3BnxnginQtD96Caq1OH1V3UMQDK-k/view?usp=sharing).
                                   [CIFAR-100](https://drive.google.com/file/d/1jmtzuefU6yzOZwIrYVzZi4TDW-0_WSxC/view?usp=sharing)
                                   [SVHN](https://drive.google.com/file/d/1f-kPfnDAzk4R4vmpcWAYSiITNN11dUX1/view?usp=sharing)

A reference FS+PD model trained on [CIFAR-10](https://drive.google.com/file/d/1WditVmllnC5Z3TzTOVzLnlJqjLeuFA9d/view?usp=sharing).
                                   [CIFAR-100](https://drive.google.com/file/d/1KuGih4Z388vzlLdmEmGjFa-hEZIixD1o/view?usp=sharing)
                                   [SVHN](https://drive.google.com/file/d/14e_vWzXoZ8ovrBexxiobd5AQSuNQUNJc/view?usp=sharing)





## Reference
Haichao Zhang and JianyuWang. Defense against adversarial attacks using feature scattering-based adversarial training. In Advances in Neural Information Processing Systems, pp. 1829–1839, 2019.
