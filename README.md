# Cats vs. Dogs

This repository contains a solution to [cats vs dogs kaggle](https://www.kaggle.com/c/dogs-vs-cats) competition. The dataset
contains about 25,000 pictures of cats and dogs and an algorithm to be developed should classify them correctly.
I used a cnn architecture similar to the AlexNet with minor adjustments.
One of the challenges is to feed the network because probably your machine is not able to load the dataset entirely. 
Further more it is not efficient at all. so I used a data generator with python to move only small batches of dataset to
the ram each time. The accuracy was confidently more than %90 on both training and test sets.

## note
```
I was running my code with a batch size equal to 64 for a long time. I observed that the accuracy bounces up and down
in each epoch and its trend is too noisy. I solved this issue by doubling the batch size to 128. You may need this conclusion
in similar problems so I decided to note it.
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details


