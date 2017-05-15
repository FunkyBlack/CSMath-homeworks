# **CSMath-homeworks** 
## 1. [Curve fitting][]

> ### Goal  
>Implement polynomial curve fitting in python and with TensorFlow (optionally).
>1. sample the function curve of y=sin(x) with Gaussian noise  
>2. fit degree 3 and 9 curves in 10 samples  
>3. fit degree 9 curves in 15 and 100 samples  
>4. fit degree 9 curve in 10 samples but with regularization term  
> ### Requirements  
>1. Programming lanuage: python  
>2. Optional package: tensorflow web  
>3. Plot the results in matplotlibIntroduction in Chinese Tutorial
### Results  
Fit degree 3 and 9 curves in 10 samples:  
![order3_num10](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework1/order3_num10.png)  
![order9_num10](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework1/order9_num10.png)  
Fit degree 9 curves in 15 and 100 samples:  
![order9_num15](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework1/order9_num15.png)  
![order9_num100](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework1/order9_num100.png)  
Fit degree 9 curve in 10 samples but with regularization term:  
![order9_num10_with_penality](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework1/order9_num10_with_penality.png)  
Degree 9 in 10 and 100 samples using Tensorflow with regularization term (20000 iterations):  
![TF1](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework1/CurveFittingTF_num10.png)  
![TF2](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework1/CurveFittingTF_num100.png)  

## 2. [PCA visualization][]  
>### Goal  
>Represent digits '3' in 2D  
>1. convert data from the UCI Optical Recognition of Handwritten Digits Data Set  
>2. perform PCA over all digit '3' with 2 components  
>3. plot the PCA results  
>### Requirements  
>1. Programming lanuage: python  
>2. Plot the results in matplotlib  
>3. Use TensorFlow if possible  
### Results  
First digit '3' in the dataset:  
![PCA1](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework2/Visual_digit_3.png)  
PCA over all digit '3' with 2 components:  
![PCA2](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework2/Digit3_using_PCA.png)  
Raw digit '3' on red dots above:  
![PCA3](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework2/Raw_digit3_on_red_dots.png)  
Visualization of mean and first 2 components of digit '3':  
![PCA4](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework2/Principal_Components.png)  

## 3. [MOG][]
>### Goal  
>Implement MOG in 2D case  
>1. Generate sampling points from a 2D Gaussian distribution  
>2. Generate sampling points from a 2D Mixture of Gaussian (MOG) distribution  
>3. Estimate parameters of generated data via step2 by using the E-M method  
>4. Plot iteration results  
>### Requirements  
>1. Programming lanuage: python  
>2. Plot the results in matplotlib  
### Results  
Using EM method to learn parameters of 2D-MOG distribution:  
![MOG1](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework3/EM_iterations.png)  
Comparison of estimated (dashed) and actual (solid) values:  
![MOG2](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework3/MOG_Using_EM.png)  
4D-MOG using EM method:  
![MOG3](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework3/EM_iterations_ndim4.png)  
![MOG4](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework3/MOG_Using_EM_ndim4.png)  

## 4. [Levenberg-Marquardt method][]
>### Goal
>Implement the Levenberg-Marquardt method  
>### Requirements
>1. Programming lanuage: python  
>2. Design your own testing functions  
### Results
Testing function is <img src="http://chart.googleapis.com/chart?cht=tx&chl=f(x)=x^{T}Ax%2Be^{-b^{T}x}%2Be^{-x^{T}x}" style="border:none;">: 
![LM](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework4/LM_iterations.png)  

## 5. [SVM][]
>### Goal
>Implement (simplified) SVM method  
>1. input 2D data and their label (in two classes)  
>2. implement quadratic programming  
>3. output (and plot) classification results
>### Requirements
>1. Programming lanuage: python  
>2. Plot the results in matplotlib
### Results
Data and label generated using [*sklearn*][]:  
![SVM1](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework5/Generated_data.png)  
SVM using linear kernel ('o' for training and 'x' for testing):  
![SVM2](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework5/SVM_visual_linear.png)  
SVM using rbf kernel:  
![SVM2](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework5/SVM_visual_rbf.png)  
SVM using polynomial kernel:  
![SVM2](https://github.com/FunkyBlack/CSMath-homeworks/raw/master/Homework5/SVM_visual_polynomial.png)  






[Curve fitting]: https://github.com/FunkyBlack/CSMath-homeworks/tree/master/Homework1
[PCA visualization]: https://github.com/FunkyBlack/CSMath-homeworks/tree/master/Homework2
[MOG]: https://github.com/FunkyBlack/CSMath-homeworks/tree/master/Homework3
[Levenberg-Marquardt method]: https://github.com/FunkyBlack/CSMath-homeworks/tree/master/Homework4
[SVM]: https://github.com/FunkyBlack/CSMath-homeworks/tree/master/Homework5
[*sklearn*]: http://scikit-learn.org/
