# Summary week 4
During this assignment, I investigated how different hyperparameters and architectural choices affect the performance of a convolutional neural network. My goal wasn't just to find the "best" model, but rather to better understand what happens when, for example, you lower the learning rate, add dropout, or change the number of filters.

I chose an iterative approach: step by step, I formulated a hypothesis, tested it through experiments, and then analyzed and visualized the results. During the process, I noticed that some models stopped training too early, making the influence of certain parameters—such as filters—difficult to see. This led me to adjust the learning rate, giving the training more room to show what the model was truly capable of.

The dataset I used consisted of images with two classes (e.g., bees vs. ants), which kept the problem manageable and clearly visible the impact of model changes. Ultimately, with this assignment I aim to provide insight into how the choices you make when building a model are reflected in the performance and learning process of a network.

### 1. First exploration: influence of architectural choices
My first experiment was an exploration of how different architectural choices affect the performance of a convolutional neural network. Instead of examining one variable at a time, I chose to test a broader combination of settings to quickly gain an initial impression of which factors have the greatest impact.

I combined the following four variables:
- the number of filters in the conv layers: 16, 32, or 64,
- the number of conv blocks in the model: 2, 3, or 4,
- the use of batch normalization: on or off,
- the use of maxpooling: on or off.

I kept all other settings constant: dropout was off, the learning rate was 1e-2, and training used early stopping (patience = 3, max 20 epochs).

Based on the theory, I expected that batch norm and maxpooling would stabilize training and improve generalization, especially for deeper models. Furthermore, I assumed that models with more filters and more conv blocks should be able to learn more complex patterns and thus achieve higher performance—as long as there was no overfitting or instability during training.

The experiments proved difficult to determine whether batch norm actually had an effect. This was mainly because the accuracy of the validation set did not improve sufficiently after just a few epochs. It was noticeable, however, that the accuracy of the validation set was higher in situations where maxpooling was used. It was also difficult to recognize a clear pattern regarding the number of filters and conv blocks. Therefore, I repeated the experiment, but with a lower learning rate (1e-3) to prevent excessive steps, and always with maxpooling.

The visualization below shows the results of these experiments. The variation in the best validation accuracy found for the different trails is shown for different numbers of filters and with or without batch normalization. It consistently shows that accuracy is higher when a batch standard is present. Here, a number of 32 filters seems to perform best, as the variance in accuracy is then lowest, but this required further refinement. After further experimentation with different filter numbers, 32 proved to yield the best results.

<img src="./images/Effect of batchnorm with different amount of filters.png" alt=" Effect of batchnorm with different amount of filters" width="400"/>

### 2. Exploring dense layers: influence of `units1` and `units2`
After exploring the key choices in the convolutional part—such as the number of filters, batch norm, and max pooling—I decided to largely fix those parameters. Only the number of conv blocks (`num_blocks`) remained flexible between 3 and 4, as previous experiments hadn't shown a clear preference in this regard.

In this series of experiments, I focused on the structure of the dense layers at the end of the network: the number of units in the first (`units1`) and second (`units2`) fully connected layers. These layers follow the convolutional layers and a global average pooling.

My hypothesis was that larger dense layers would increase the model's capacity to model more complex decision boundaries, especially if the convolutional part already provides good representations. At the same time, I knew this could also lead to overfitting, but because dropout wasn't yet enabled (`dropout_dense_rate = 0.0`), I could isolate the effect of just the number of units.

The other settings in this experiment were:
- `filters=32`, `batchnorm=True`, `maxpooling=True`
- `num_blocks ∈ {3, 4}`
- `dropout_conv_rate=0.0`, `dropout_dense_rate=0.0`
- `lr=1e-3`, `patience=3`

The visualization below shows the achieved validation accuracies for different combinations of `units1` and `units2`. The y-axis shows the highest validation accuracy per trail, and the color indicates the number of units in the second dense layer. This shows that adding more units to the first full layer doesn't improve the results. The number of units in the second layer is, however, still somewhat uncertain, as there is hardly any difference between 32 or 64 units. Furthermore, it turned out that three conv-blocks were more suitable.

<img src="./images/Effect of different amount of units.png" alt=" Effect of different amount of units" width="400"/>

### 3. Experimenting with Dropout Rate
The previous phase revealed that the number of units in the second dense layer was still uncertain, as there was hardly any difference between 32 and 64 units. Furthermore, it emerged that three conv-blocks would be more suitable.

To gain more clarity on this, I further experimented with the number of units in the dense layers, particularly with "units2." I also re-examined the number of filters to see if this made the effects of the units more apparent. As a final step, I added the dropout rate to the experiments, with values ​​ranging between 0.0 and 0.5, to investigate the effect of regularization.

Although these latest experiments have given me more insight into the effects of units, filters, and dropout, it remained difficult to draw a clear conclusion about the best configuration. Accuracy was fairly homogeneously distributed, resulting in no clear winner (see the figure below). This suggests that other factors may be at play, or that the model and data offer limited room for improvement within the parameter range examined.

<img src="./images/Effect of different amount of units and dropout rate.png" alt=" Effect of different amount of units and dropout rate" width="400"/>
<img src="./images/Effect of different amount of filters and dropout rate.png" alt=" Effect of different amount of filters and dropout rate" width="400"/>


### Conclusion
While the results do not yield a theoretically best configuration, experimenting with different architecture and regularization parameters has yielded valuable insights into their influence on the learning process. The relatively probable distribution of the accuracy points to the model potentially reaching a performance ceiling for this dataset and setup.

Since the dataset is relatively small, using a pre-trained model (transfer learning) in future work could provide more clarity and better performance. Further research with larger datasets and more complex architectures, such as skip connections, could also help to better explore the model's limits.

### References & links
- Assignment description: [Link to the assignment](./instructions.md)
- Hypertuning script: [run_hypertune.py](./scr/run_hypertune.py)
- Notebook with experiments: [Experiments notebook](./hypertune_notebook.ipynb)


[Go back to Homepage](../README.md)