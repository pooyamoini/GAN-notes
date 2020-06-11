
# Generative Adversial Networks





## Maximum of objective function relative to the discriminator
We assume that the capacity of our model is unlimited and we wanna find the discriminator in which the objective function be its maximum. So we just need to the derivative of the objective relative to the discriminator be zero.

$\mathbb{V}(G, D) = \mathbb{E}_{x \sim p_{data}} [\log \mathbb{D}(x)] + \mathbb{E}_{x \sim p_{G}} [\log(1 - \mathbb{D}(x))] \rightarrow \frac{d\mathbb{V}(G, D)}{dD} = \frac{d}{dD}\int_x p_{data} [\log \mathbb{D}(x)] + p_G [\log(1-\mathbb{D}(x))] dx\\
\int_x \frac{p_{data}}{\mathbb{D}(x)} - \frac{p_G}{1 - \mathbb{D}(x)} dx = 0 \Rightarrow \frac{p_{data}}{\mathbb{D}(x)} - \frac{p_G}{1 - \mathbb{D}(x)} = 0 \longrightarrow \frac{p_{data}}{\mathbb{D}(x)} = \frac{p_G}{1 - \mathbb{D}(x)} \Longrightarrow \mathbb{D}(x) = \frac{p_{data}}{p_G + p_{data}}$

Now we want to put the optimal discriminator into the objective function and see what's gonna happen.

$\mathbb{D^*}(x) = \frac{p_{data}}{p_{data} + p_G} \, , \mathbb{V}(G, D) = \mathbb{E}_{x \sim p_{data}} [\log \mathbb{D}(x)] + \mathbb{E}_{x \sim p_{G}} [\log(1 - \mathbb{D}(x))] \\ $
$\mathbb{V}(G, D^*) = \mathbb{E}_{x \sim p_{data}} [\log (\frac{p_{data}}{p_{data} + p_G})] + \mathbb{E}_{x \sim p_{G}} [\log(1 - (\frac{p_{data}}{p_{data} + p_G}))] =  \mathbb{E}_{x \sim p_{data}} [\log (\frac{p_{data}}{p_{data} + p_G})] + \mathbb{E}_{x \sim p_{G}} [\log(\frac{p_G}{p_{data} + p_G})] $
$\mathbb{V}(G, D^*) = \int_x{ p_{data} \log{[\frac{p_{data}}{\frac{p_{data} + p_G}{2}}]} + p_G \log{[\frac{p_G}{ \frac{p_G + p_{data}}{2} }]} dx} - 2\log{2}$
$\longrightarrow \mathbb{V}(G, D^*) = -2\log{2} + \mathbb{KL}(p_{data} || \frac{p_{data} + p_G}{2}) + \mathbb{KL}(p_G || \frac{p_{data} + p_G}{2}) = -2\log{2} + 2\mathbb{JS}(p_{data}||p_q)$

As we concluded , optimizing $\mathbb{V}(G, D)$ where $D = D^*$ is equivalent to minimize the Jensen–Shannon distance of $p_{data}$ and $p_G$. But all these results is in theory, in practice we can't assume that the capasity is unlimited. So in practice during a batch we update parameters of both discriminator and generative multiple times.

## Gradient problem

Assume that the dataset domain and generative domain do not overlap. Also assume that $D$ is so near to $D^*$. 

We are going to calculate gradient of $\log(1-D(x))$ relative to network's logit. Discriminator is equivalent to $\sigma(a)$ which $a = {f}(x)$. In this situation $a$ is the logit.

$\mathbb{F} = \log(1 - D(x)) = \log(1 - \frac{1}{1 + \exp(-a)}) = \log(\frac{\exp(-a)}{1+\exp(-a)}) = -a - \log(1 + \exp(-a))$

$\rightarrow \nabla_a \mathbb{F} = -1 + \frac{\exp(-a)}{1 + \exp(-a)} = \frac{-1}{1 + \exp(-a)} = -\sigma(a)$

So whats the problem? <br/> 
If discriminator is near to the optimized, it will return 0 for the generative samples. So we have $\sigma(a) = D(x) = 0$. Therefore In accordance with the results we concluded, the gradient will be zero too. So the gradient won't be reached to the previous layers.<br/>
As the solution we suggest that use $-\log{D(x)}$ as the loss function.

$\mathbb{F} =  -\log{D(x)} = -\log{ \frac{1}{1+\exp(-a)} } = \log{(1 + \exp(-a))}$ <br/>
$\longrightarrow \nabla_a \mathbb{F} = -\frac{\exp(-a)}{1 + \exp(-a)} = \sigma(a) - 1$

So in this state, the gradient will be $\, -1$ and the problem is solved.

## MLE-GAN

In this section, we're gonna analyze another approach we be equivalent to MLE approach. <br/>
In MLE approach, the loss function for generative is:
$$\mathcal{L}_{MLE}(\theta) = \mathbb{E}_{x \sim p_{data}}[-\log p_G(x)]$$
which $\theta$ is the parameters of the generative network.<br/>
In our new approach the loss function for the discriminator wont change but the loss function for the generative is :
$$\mathcal{L}_{\small{MLE-GAN}}(\theta) = \mathbb{E}_{x \sim p_G} [\mathcal{f}(x)]$$
Now we want to calculate the function $\mathcal{f}$. <br/>
We have to find the fuction $\mathcal{f}$ which gradients of $\mathcal{L}_{\small{MLE-GAN}}$ respect to $\theta$ be equal to gradients of $\mathcal{L}_{\small{MLE}}$ respect to $\theta$.<br/><br/>
$\nabla_{\mathcal{\theta}} \mathcal{L}_{\small{MLE}}(\theta) = \mathbb{E}_{x \sim p_{G}}[- \nabla_{\theta}\log p_G(\mathcal{x})] \quad , \quad \nabla_{\theta} \mathcal{L}_{MLE-GAN}(\theta) = \nabla_{\theta} \int \mathcal{f}(x) \mathcal{p}_{G}(x) dx = \int_x \nabla_{\theta}p_G(x) \, f(x) dx$ <br/>
$ \rightarrow \nabla_{\theta} \mathcal{L}_{MLE-GAN}(\theta) = \int_x \frac{\nabla_{\theta}p_G(x)}{p_G(x)} f(x) p_G(x) dx = \mathbb{E}_{x \sim p_G}(\nabla_{\theta}[\log p_G(x)] f(x))$ <br/>
$$\Longrightarrow f = -\frac{p_{data}(x)}{p_G(x)}$$
$D^* = \frac{p_{data}(x)}{p_{data}(x) + p_G(x)} \longrightarrow f = -\frac{D^*}{1 - D^*} = -\exp(a)$


```
}
```
