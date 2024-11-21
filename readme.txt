1.if you want train the model, instead train from very begin, you can load the "unet.pt" and train under the pre-trained one, 
   it's been trained for 10 epoches already, and the trained model can give some distinguishable result already (not very good though).
2.picture need the fixed size 1024 * 768 (no matter train or test)
3.the test output will be an original image and its mask image (display in sequence)
