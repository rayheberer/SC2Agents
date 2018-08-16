# A2CAtari Results

## Model Details

### Inputs/Outputs
This model takes as input all screen feature layers, all minimap feature layers, and the flat `player` features. At any given step, it can output any available action (function identifier) with any set of possible arguments.

### Estimator
Categorical screen and minimap (spatial) features are first embedded into continuous space through one-hot encoding in the channel dimension followed by a 1x1 convolution, and numeric features are scaled by a log-transform. Spatial features are then processed by their own copy of a two-layer convolutional neural network with 16 and 32 filters, 8x8 and 4x4 kernel dimensions, and strides of 4x4 and 2x2 respectively, all with ReLU activations. Flat features are processed by a 64 unit fully-connected layer with a tanh activation.

The outputs from the final convolutional layers and the fully-connected layer are flattened and concatenated, before the final state representation is obtained through a fully-connected layer with 256 units and a ReLU activation.

The value estimate is obtained by feeding the state representation vector into a single linear unit.

Policies for function identifiers (`Move_screen`, `No_op`, etc.) and for function arguments (`queue`, `now`, `(x, y)`, etc.) are produced independently by fully connected layers that take the state representation as input and have a softmax activation.

Conceptually, the network can be thought of as two models, a value estimator and a policy estimator, that share most of their weights (all except those in the output layers).

### Policy
At each step, a function identifier is sampled from the corresponding output layer, where the values of the output units together form the probability distribution of actions given the state fed in. First, unavailable function identifiers (e.g. moving a unit when no unit is selected) are masked out and the nonzero outputs are then renormalized.

Depending on the type of function identifier sampled, one or more function arguments may need to be sampled from their corresponding output layers.

The action the agent takes is then the function identifier together with all of its potential arguments.

The A2C agent is on-policy, meaning that the policy it uses to generate experiences while training is the same policy being directly optimized.

### Training Procedure
After `trajectory_training_steps` have elapsed, or upon receiving a signal indicating a step is a terminal one, the weights of the value/policy estimator are updated through a gradient descent optimization algorithm (RMSProp).

The agent maintains double-ended queues of previous states observed, actions performed, and rewards received. At most `trajectory_training_steps` most recent experiences are stored. At training time, these form a batch and n-step returns are calculated for each step to be provided as targets for the network. In general, an n-step return gives the discounted cumulative reward from the current state up to the futuremost state where the reward is explicitly known, at which point the return is bootstrapped off of the network's value estimate for that state. A full-sized batch will then contain a 1-step return, a 2-step return, and so on up to a `trajectory_training_steps`-step return for the oldest experience stored.

The A2C gradient has 3 terms: a policy gradient, a value estimation gradient, and an entropy regularization term. Roughly speaking, by performing gradient descent (or ascent, depending on the sign) on the A2C gradient, the model weights are optimized towards policies with greater expected returns, and more accurate value estimates, with the regularization term encouraging exploration. 

The hyperparameters `value_gradient_strength` and `regularization_strength` scale the contribution of each term in the gradient towards the overall gradient. The `learning_rate` scales the overall size of the update steps taken. Finally, the policy and value estimation gradients are scaled by the "advantage", which is the difference between the target return and the estimated value for an observed state, so that states with large positive advantages (observed to yield higher returns than expected) as well as those with large negative advantages (observed to yield lower returns than expected) will contribute more to the gradient updates.

The gradients for each of the steps in a batch are accumulated (summed) before being applied to the weights.  

## Results
<table align="center">
  <tr>
    <td align="center"></td>
    <td align="center">Test Score</td>
    <td align="center">Training Episodes</td>
    <td align="center">Notes</td>

  </tr>
</table>

## Training Notes/Caveats
* There are a number of flat features that this agent does not take as input.