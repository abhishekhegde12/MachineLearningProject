      ** Implementation of PCA with ANN algorithm for Face recognition**

  **Steps involved in training:**
1. Generate the face database:
          Each face image is represented in the form a matrix having m rows and n columns, 
          where each pixel (x,y) such that xm, and yn shows pixel location of the image 
          as well as the direction.
          For the simplicity we are assuming each face image as a column vector, if we have 
          p images then the size of the face database will be mn*p.
          Let’s say face database is denoted as (Face_Db)𝑚𝑛∗𝑝
2. Mean Calculation:
          Calculate the mean of each observation
          M𝑖=∑ ∑ 𝐹𝑎𝑐𝑒_𝐷𝑏(𝑖,𝑗)
          𝑝
          𝑗=1
          𝑚𝑛
          𝑖=1
          here mean vector will have the dimension (M)𝑚𝑛∗1
3. Do mean Zero 
          Subtract mean face from each face image, let’s say this mean zero face data as 
          , and calculated as
          ( (i))𝑚𝑛∗𝑝= (Face_Db(i))𝑚𝑛∗𝑝- (M)𝑚𝑛∗1
          Where i1,2,3 ……., p.
4. Calculate Co-Variance of the Mean aligned faces ()
          Here there is a slightly variation in calculation of covariance, generally we prefer 
          to calculate the covariance of data by:
          C = ∑ (𝑋𝑖 − 𝑋) ((𝑌𝑖 − 𝑌))
          𝑛 𝑡
          𝑖=1
          Where 𝑋, 𝑌 𝑎𝑟𝑒 𝑡ℎ𝑒 𝑚𝑒𝑎𝑛 𝑜𝑓 𝑋𝑖 𝑎𝑛𝑑 𝑌𝑖
          , and C is the covariance matrix. If we will 
          follow the same convention on face data we will get.
          C (mn,mn) =∑ ∑ ∑ ((z, i) − 𝑀𝑧,𝑖) ∗ ((z, i) − 𝑀𝑦,𝑖)
          𝑡
          𝑝
          𝑖=1
          𝑚𝑛
          𝑦=1
          𝑚𝑛
          𝑧=1
          Here we will get mn direction, which is very hard to compute, store and process. 
          It also increases the program complexity, hence in 1991 Turk and Peterland [1] two 
          researches suggested a new way to calculate the co-variance that is basically 
          known as surrogate covariance, that is:
          C (p,p) =∑ ∑ ∑ ((z, i) − 𝑀𝑧,𝑖) ∗ ((y, i) − 𝑀𝑦,𝑖)
          𝑚𝑛
          𝑖=1
          𝑝
          𝑦=1
          𝑝
          𝑧=1
          Hence here will get only p * p dimension, which is easy to compute and process, 
          the idea behind computing the surrogate covariance suggested by turk and 
          peterland that, these are only the valid direction where we will get maximum 
          variances, and rest of the directions are insignificant to us. Means these are 
          direction where we will get the eigenvalues and for rest we will get eigenvalues 
          equal to zero. 
5. Do eigenvalue and eigenvector decomposition:
          Now we have covariance matrix 
          (C)𝑝∗𝑝, 𝑓𝑖𝑛𝑑 𝑜𝑢𝑡 𝑡ℎ𝑒 𝑒𝑖𝑔𝑒𝑛𝑣𝑒𝑐𝑡𝑜𝑟𝑠 𝑎𝑛𝑑 𝑒𝑖𝑔𝑒𝑛𝑣𝑎𝑙𝑢𝑒𝑠. Let we have eigenvector 
          (V)𝑝∗𝑝 𝑎𝑛𝑑 𝑒𝑖𝑔𝑒𝑛𝑣𝑎𝑙𝑢𝑒𝑠 ()𝑝∗𝑝.
6. Find the best direction (Generation of feature vectors)
          Now select the best direction from p directions, for this sort the eigenvalues in the 
          descending order and decide a k value, which represents the number of selected 
          eigenvectors to extract k direction from all p direction. On the basis of k value we 
          can generate the Feature vector ()𝑝∗𝑘.
7. Generating Eigenfaces:
          For generating the eigenfaces () project the each mean aligned face to the 
          generated feature vector.
          ()𝑘∗𝑚𝑛 = ()
          𝑡
          𝑘∗𝑝
          * ()
          𝑡
          𝑝∗𝑚𝑛
8. Generate Signature of Each Face:
          For generating signature of each face (), project each mean aligned face to the 
          eigenfaces.
          ()𝑘∗𝑖= ()𝑘∗𝑚𝑛 ∗ ()𝑚𝑛∗𝑖
          Where i  1, 2, 3, ….., p. hence  will have the size k * p.
9. Apply ANN for traning:
      After getting the best eigen vector apply back propagation neural network as 
      discussed in the video.
   
**Steps involved in Testing:**
1. For testing, let we have an image (I), make it as a column vector say (I)
    1
    𝑚𝑛∗1
    2. Do mean Zero, by subtracting mean face (M) to this test face, say it (I)
 
2
    𝑚𝑛∗1
    = (I)
    1
    𝑚𝑛∗1
    − (𝑀)𝑚𝑛∗1

3. Project this mean aligned face (I)
2
to eigenfaces (), we will get the projected 
test face ().
()𝑘∗1 = ()𝑘∗𝑚𝑛 ∗ (I)
2
𝑚𝑛∗1
4. Now we have projected test face  and signature of each face , use the trained 
ANN model to predict the unknown face.
Take 60% data as training set and 40 % data as test set, evaluate your classifier on the 
following
Factors:
a) Change the value of k and then, see how it changes the classification accuracy. Plot a 
graph between accuracy and k value to show the comparative study.
b) Add imposters (who do not belong to the training set) into the test set and then
recognize it as the not enrolled person
