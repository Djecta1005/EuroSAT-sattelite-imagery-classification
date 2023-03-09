# SpaceSenseChallenge

Here is an implementation of image classification for on the EuroSAT land cover classification dataset. 

The model architecture that we have used is Resnet 101.

## Dataset:
After considering my familiarity with the frameworks, I decided to utilize Pytorch for my project. To ensure seamless integration with Pytorch pipelines, such as multithreaded data loaders, transform operations, samplers, etc., I chose to implement a custom loader by subclassing torchvision ImageFolder. As EuroSAT lacks a defined test set, I generated one using a 90 % / 10 % split with a fixed random seed to maintain consistency. 

## Training:
During the training phase, I implement some data augmentation techniques to improve the model's performance, including random horizontal and vertical flips using the torchvision utilities. To optimize my model's accuracy, I utilize a pretrained resnet101 model, replacing its head as I have only 10 classes. By default, only the head is fine-tuned during the training process. 


# Constraints of the current solution:

- **Limited dataset size:** As the dataset consists of only 27,000 images, the model's accuracy may not generalize well to other similar datasets with a larger number of images.
- **Limited variety of classes:** The dataset includes only 10 different classes of land use, limiting the model's ability to recognize more complex patterns in a more diverse set of classes.
- **Limited data augmentation:** While some data augmentation techniques, such as random horizontal and vertical flips, are implemented during training, more advanced techniques could be utilized to further enhance the model's performance.
- **Pretrained model limitations:** The use of a pretrained resnet101 model with its head replaced may not be the best option for this specific task, as the dataset is small and not necessarily similar to the dataset used to pretrain the model.
- **Limited hyperparameter tuning:** The current solution employs default hyperparameters for training, and a more thorough optimization could lead to higher accuracy.
# Potential improvements to the solution:

- **More advanced data augmentation:** Implementing more advanced data augmentation techniques, such as rotation, scaling, and translation, could improve the model's ability to recognize patterns and reduce overfitting.
- **Fine-tune pretrained models:** Fine-tuning models specifically for this dataset, or training models from scratch, could lead to improved performance compared to using a generic pretrained model.
- **Hyperparameter tuning:** Conducting a thorough search for optimal hyperparameters could help to maximize the model's performance.
- **Implement more advanced optimization techniques:** Utilizing techniques such as learning rate scheduling, weight decay, or gradient clipping could help to improve the model's training and generalization.
- **Utilize more advanced architectures:** Implementing more advanced neural network architectures, such as attention mechanisms or capsule networks, could lead to improved performance on this specific task.
