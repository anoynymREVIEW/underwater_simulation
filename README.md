# Physics Informed and Data Driven Simulation of Underwater Images via Residual Learning

## Data
* [NYU Depth V2 (50K)](https://tinyurl.com/nyu-data-zip) (4.1 GB): You don't need to extract the dataset since the code loads the entire zip file into memory when training. We have directly use the same dataset and have followed the same protocol, mentioned [here](https://github.com/ialhashim/DenseDepth)
* [Make-3D](http://make3d.cs.cornell.edu/data.html): We have used same experimental protocol, mentioned [here](https://github.com/shirgur/UnsupervisedDepthFromFocus). Please note that we have used only **Dataset-1** from Make-3D dataset.

<br/>
<br/>

To generate the ground truth data of underwater images, see :

### **NYU dataset** :
      
⮕ See the file : /Technique_1/**generate_save_main.py** <br/>
⮕ See the file : /Technique_1/**generate_save_main_ver1.py**<br/>

### **Make-3D dataset** :
      
⮕ To generate the training GT data : /Technique_1/**generate_save_make_3D.py** <br/>
⮕ To generate the testing GT data : /Technique_1/**generate_save_make_3D_test.py**<br/>

<br/>
<br/>
The following algorithms are implemented and tested

## Technique 1
Look into the Technique 1 folder to see the code 
<br/>
<br/>
### **Training with NYU dataset** :
      
⮕ **Technique 1 :** See the file for training : /Technique_1/**train_3_1.py** <br/>
⮕ **Technique 1 Variant 1 :** See the file for training : /Technique_1/**train_3_1_weight_1.py**<br/>
⮕ **Technique 1 Variant 2 :** See the file for training : /Technique_1/**train_3_1_weight_2.py**

### **Testing with NYU dataset** :
      
⮕ **Technique 1 :** See the file for testing : /Technique_1/**perform_test_1.py** <br/>
⮕ **Technique 1 Variant 1 :** See the file for testing : /Technique_1/**perform_test_1_weight_1.py**<br/>
⮕ **Technique 1 Variant 2 :** See the file for testing : /Technique_1/**perform_test_1_weight_2.py**
<br/>
<br/>
### **Training with Make-3D dataset** :
      
⮕ **Technique 1 :** See the file for training : /Technique_1/**train_3_1_Make_3D.py** <br/>
⮕ **Technique 1 Variant 1 :** See the file for training : /Technique_1/**train_3_1_weight_1_Make_3D.py**<br/>
⮕ **Technique 1 Variant 2 :** See the file for training : /Technique_1/**train_3_1_weight_2_Make_3D.py**

### **Testing with Make-3D dataset** :
      
⮕ **Technique 1 :** See the file for testing : /Technique_1/**perform_test_1_make_3D.py** <br/>
⮕ **Technique 1 Variant 1 :** See the file for testing : /Technique_1/**perform_test_1_weight_1_make_3D.py**<br/>
⮕ **Technique 1 Variant 2 :** See the file for testing : /Technique_1/**perform_test_1_weight_2_make_3D.py**

<br/>
<br/>

## Technique 2
Look into the Technique 2 folder to see the code 
<br/>
<br/>
### **Training with NYU dataset** :
      
⮕ **Technique 2 :** See the file for training : /Technique_2/**train_4.py** <br/>
⮕ **Technique 2 Variant 1 :** See the file for training : /Technique_2/**train_4_weight_1.py**<br/>
⮕ **Technique 2 Variant 2 :** See the file for training : /Technique_2/**train_4_weight_2.py**

### **Testing with NYU dataset** :
      
⮕ **Technique 2 :** See the file for testing : /Technique_2/**perform_test_1.py** <br/>
⮕ **Technique 2 Variant 1 :** See the file for testing : /Technique_2/**perform_test_1_weight_1.py**<br/>
⮕ **Technique 2 Variant 2 :** See the file for testing : /Technique_2/**perform_test_1_weight_2.py**
<br/>
<br/>
### **Training with Make-3D dataset** :
      
⮕ **Technique 2 :** See the file for training : /Technique_2/**train_4_Make_3D.py** <br/>
⮕ **Technique 2 Variant 1 :** See the file for training : /Technique_2/**train_4_weight_1_Make_3D.py**<br/>
⮕ **Technique 2 Variant 2 :** See the file for training : /Technique_2/**train_4_weight_2_Make_3D.py**

### **Testing with Make-3D dataset** :
      
⮕ **Technique 2 :** See the file for testing : /Technique_2/**perform_test_1_make_3D.py** <br/>
⮕ **Technique 2 Variant 1 :** See the file for testing : /Technique_2/**perform_test_1_weight_1_make_3D.py**<br/>
⮕ **Technique 2 Variant 2 :** See the file for testing : /Technique_2/**perform_test_1_weight_2_make_3D.py**


<br/>
<br/>

## Technique 3
Look into the Technique 2 folder to see the code 
<br/>
<br/>
### **Training with NYU dataset** :
      
⮕ **Technique 3 :** See the file for training : /Technique_3/**train_4.py** <br/>
⮕ **Technique 3 Variant 1 :** See the file for training : /Technique_3/**train_4_weight_1.py**<br/>
⮕ **Technique 3 Variant 2 :** See the file for training : /Technique_3/**train_4_weight_2.py**

### **Testing with NYU dataset** :
      
⮕ **Technique 3 :** See the file for testing : /Technique_3/**perform_test_1.py** <br/>
⮕ **Technique 3 Variant 1 :** See the file for testing : /Technique_3/**perform_test_1_weight_1.py**<br/>
⮕ **Technique 3 Variant 2 :** See the file for testing : /Technique_3/**perform_test_1_weight_2.py**
<br/>
<br/>
### **Training with Make-3D dataset** :
      
⮕ **Technique 3 :** See the file for training : /Technique_3/**train_4_Make_3D.py** <br/>
⮕ **Technique 3 Variant 1 :** See the file for training : /Technique_3/**train_4_weight_1_Make_3D.py**<br/>
⮕ **Technique 3 Variant 2 :** See the file for training : /Technique_3/**train_4_weight_2_Make_3D.py**

### **Testing with Make-3D dataset** :
      
⮕ **Technique 3 :** See the file for testing : /Technique_3/**perform_test_1_make_3D.py** <br/>
⮕ **Technique 3 Variant 1 :** See the file for testing : /Technique_3/**perform_test_1_weight_1_make_3D.py**<br/>
⮕ **Technique 3 Variant 2 :** See the file for testing : /Technique_3/**perform_test_1_weight_2_make_3D.py**

<br/>
<br/>

## Encoder-Decoder Direct 
Please see the [Encoder_Decoder_Direct](./Encoder_Decoder_Direct/README.md) directory of this repository for details on how to train and evaluate our method.

## UNIT 
Please see the [UNIT](./UNIT/README.md) directory of this repository for details on how to train and evaluate our method.

## MUNIT 
Please see the [MUNIT](./MUNIT/README.md) directory of this repository for details on how to train and evaluate our method.

## Pix2Pix_GAN 
Please see the [Pix2Pix_GAN](./Pix2Pix_GAN/README.md) directory of this repository for details on how to train and evaluate our method.

## CycleGAN 
Please see the [CycleGAN](./CycleGAN/README.md) directory of this repository for details on how to train and evaluate our method.

## DRIT 
Please see the [DRIT](./DRIT/README.md) directory of this repository for details on how to train and evaluate our method.
