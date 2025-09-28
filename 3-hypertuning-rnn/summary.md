# Summary week 3
The goal of this assignment is to improve the RNN model to an accuracy of at least over 90%. The provided gesture dataset was used for this purpose. GRU, LSTM, and Conv1d layers were used during the experiments in the RNN model. For more details about the assignment, see the [instructions](./instructions.md).  

### Experiments
First, the baseline model is considered, followed by adjustments to it. The [notebook](./notebook.ipynb) provides a detailed overview of the code used.

#### Baseline model
The baseline model uses a single-layer GRU with 64 hidden units. It can be seen that after 10 epochs, the loss continues to decrease. The accuracy after 10 epochs is 80%. Therefore, I decided to run another 10 epochs. The accuracy score then clearly shows that the 5-point patience had an effect here. After 14 epochs, a decrease occurs, which later increases again. This results in an accuracy score of 95% after 20 epochs. Reducing the number of hidden units decreases the accuracy and therefore underfits the data. Increasing the number of hidden units to 128 results in slightly higher accuracy, but when looking at the loss, there is clearly overfitting. The loss is approximately three times greater for the test set than for the train set.  

#### Replace GRU with LSTM
Next, I experimented by using an LSTM layer instead of a GRU layer. This was to investigate whether longer dependencies are influential. The results show that 1-layer LSTM only produces comparable results after about 25 epochs. However, the loss of both the train and test sets decreases more slowly. At 20 epochs, the accuracy is only 86%. Next, the effect of multiple layers on the LSTM was examined. Of the experiments with 2, 3, or 4 layers, the 3-layer model performed best. However, this model took about 1.5 times as long to complete.

#### Increase GRU layers
Furthermore, experiments were conducted with adding extra layers for the GRU. The results showed that this resulted in overfitting. For both two and three layers, the test loss was significantly higher than that of the train set. Furthermore, the time required also increased significantly.

#### Added Conv1d before GRU
When adding a conv1d layer, it's clear that a patience of 5 is too high. This makes the difference between the train and test loss too large, leading to overfitting. Lowering the patience to 2 eliminates this problem. Varying the output of the conv1d layer shows that more outputs do not improve accuracy. For example, 16 outputs perform better than 32 or 64. However, the train loss is slightly lower than that of the test loss. Furthermore, it's clear that a combination of a conv1d layer and a 2-layer gru yields comparable accuracy, but the train set's loss is slightly lower than that of the test set. In that case, the accuracy is 95%. When 2-layer GRU is used, an output of 32 for the conv1d layer even improves accuracy, achieving 98%.  

#### Added Conv1d before LSTM
Given the positive effects of adding a Conv1d layer to GRU, adding one to LSTM was also considered. Here too, a patience of 5 is too large and causes overfitting. With a patience of 2, overfitting still occurs with 16 outputs of the conv1d layer, but not with 32 or 64. However, it can be seen that 64 outputs does not contribute to an improvement in accuracy. For LSTM, a combination of one conv1d layer followed by a two-layer LSTM also yields the best results, with a train loss of 0.0614, a test loss of 0.0890, and an accuracy of 98%. Compared to Conv1d combined with GRU, it is even slightly faster.  

### Final model and conclusion
This experiment shows that a Conv1d layer for the rnn stage improves the model's performance by determining local features. This layer, combined with LSTM or GRU, both achieved an accuracy of approximately 98%. For both, the best performance was achieved with 32 outputs from the Conv1d layer and two layers for the rnn section. Furthermore, a patience of 2 and a hidden size of 64 were used.

[Go back to Homepage](../README.md)
