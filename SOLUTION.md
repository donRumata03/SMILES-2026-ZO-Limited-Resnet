# Reproducing

Install pip dependencies and run:

```bash
pip install -r requirements.txt

python validate.py --data_dir ./data --batch_size 16 --n_batches 512 --output results.json
```

Some of the hyperparameters are configured through `zo_config.yaml`. The submitted configuration is:

```yaml
init_method: lda
use_lora: true
lora_rank: 16
ridge_alpha: 10
top_p: 0.9
```

_The obtained accuracy is: $0.6001$._

# Method

The method consists of two parts: head initialization and zero-order optimization.

## Initialization

For initialization, I used a balanced subset of train: 81 examples per class, 8100 images total which is within the sample budget.

The tested initialization methods are:

- **LDA**. Frozen backbone features are extracted for the selected training images. Then `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` is fitted with uniform class priors. The learned model is linear which is what we need because it allows to copy the learned coefficients and intercepts to the final linear layer.
This is a generative linear classifier which estimates class means in feature space and a shared covariance matrix (we have some sample to try to estimate it but not to have individual cov matrices for all the classes). Shrinkage regularizes it towards a better conditioned one to make inversion more stable (we don't have much data, so this prior-like thing is what we need). This was the best initialization: `0.5988` init top-1.

- **Ridge classifier**. Frozen features are used to fit `RidgeClassifier` and put the learned result into the last model's layer. The best tested variant, `alpha=10`, reached `0.5930` init top-1 which is nearly the same but a bit smaller.

- **ImageNet-head top-p classes**. The idea is to reuse the contents of the ImageNet head: for each class from CIFAR, use the average of the rows of imagenet classes that presumably cover this particular class. But to make it «fair», the class relation is infered purely from train data. For every selected training image, I computed the original 1000-dimensional logits of the pretrained ImageNet head. The association score between CIFAR class `c` and ImageNet class `j` was `mean_imagenet_logit[c, j] - global_mean_imagenet_logit[j]`. Negative scores were clipped to zero. Then ImageNet classes were sorted by this positive association score, and the smallest «top-p» prefix was selected where p is the fraction of the sum of all the scores of the taken rows (p is not the real softmax probability, but it's still the top-p vibe). This reached `0.3160` init top-1 and `0.3374` final top-1.

- **Zero initialization**. Used to evaluate if LoRA (see below) would be able to work without a strong initialization. The results is that the best final top-1 was `0.0181`.


## Zero-order optimization

The gradient estimator itself seems reasonable as is the optimizer — SGD — but the unreasonable part is the number of parameters learned by zero order method — ≈51k.

So I tried to decrease it with LoRA correction on the head weight:

$$
W = W_0 + A B
$$

where $W_0$ is the initialized weight, $A$ is optimized, and $B$ is a fixed random basis from $\cal{N}(0, 1)$ scaled by $1 / \sqrt{512}$.

I also tried the full version where both low-rank factors were learnable, but it was probably unstable and the quality was near-zero. The fixed-basis version that works only in random subspace is believed to have a simpler optimization landscape and it worked for me.

The gains from lora are moderate, especially for the already good LDA, but as we say, it's a fair job: it turns 0.5988 initialization into 0.6001 final result.
