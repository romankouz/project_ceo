# project_ceo

# Step-by-step Instructions for Use
1. Clone this repository.
2. Open a terminal and navigate to the root directory (.../project_ceo).
3. Run the following: `python train.py --dataset "MNIST" --optim "Adam"` 

   3a. The choice of optimizers at the moment include `"Adam"`, `"SGD"`, `"Adamax"`, `"RMSProp"` `"Adabelief"`, and `"Adagrad"`.
      The optim argument is case sensitive.
      Additionally, the choice of "All" will train all of the optimizers at once. 
      
   3b. The choice of datasets include `"MNIST"` and `"CIFAR10"`. If time permits, a setting will be implemented such that
       if the dataset exists in `torchvision.datasets`, the code will run with the optimizers, though optimizer conditions
       or convergence will not be guaranteed.
      
      
# Viewing Results
1. The training and validation accuracies are available in the /images directory.

The final paper generated for the project can be read here: [606_Project.pdf](https://github.com/roromaniac/project_ceo/files/6363608/606_Project.pdf)


