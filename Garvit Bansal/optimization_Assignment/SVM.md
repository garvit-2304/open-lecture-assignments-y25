# Inside the Mind of an SVM: From Hyperplanes to Dual Problems

This post will deal with Support Vector Machines (SVM) and the mathematics behind them. SVMs are one of the best supervised learning algorithms used in classification tasks. We will deal with the key concepts to understand the black box of SVM, starting with margins and separating data with large gaps. Then we will see the optimal margin used for maximizing, and this will lead us to Lagrange duality and Karush-Kuhn-Tucker (KKT) conditions for the solution of our optimization problem.

## MARGINS: intuition

In this section, we will try to understand the intuition behind margins.
Let’s take a binary classification problem shown in Figure 1, where white represent positive training data and black represent negative training data, and we want to find a decision boundary (line H3 this line is called a separating hyperplane ) that will separate these two data sets with high confidence such that the margin between the points and data is large(line H1 an H2 are not able to separate the data with high margin value thus represent bad “fit” to our training data ).

![Figure 1: Three potential separating hyperplanes for a classification problem.](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Svm_separating_hyperplanes_%28SVG%29.svg/500px-Svm_separating_hyperplanes_%28SVG%29.svg.png)

## Notations :

Since it’s a classification problem, we will consider a linear classifier for a binary classification problem.
y will represent labels and x will represent features.
$y=\{-1,1\}$ for class labels and
$h_{w,b}=g(w^Tx+b)$
Here $g(z)=1$ if $z \ge 0$ and $g(z)=-1$ if $z \le 0$

### Functional margins:

We define the functional margins of $(w,b)$ with respect to $(x_i,y_i)$ as
$\gamma_i=y_i(w^Tx_i+b)$
If $y_i=1$ Then, for our prediction to be correct $w^Tx_i+b >0$ and similarly if $y_i=-1$
$w^Tx_i+b < 0$ will give correct predictions. Combining this, we can say that for correct predictions $\gamma_i>0$. So functional margin is used to tell whether a point is correctly predicted or not.

If we notice that replacing w with 2w and b with 2b increases the length of the functional margin by a factor of 2. This implies that our functional margin is not scale-invariant. It might make sense to put some normalization on the functional margin.

### Geometric margins:

The geometric margin is the Euclidean distance between a certain data point and the hyperplane.
We know the distance from a line $w^Tx_i+b=0$ is given by
$\gamma_i=\frac{w^Tx_i+b}{||w||}$
Where $||w||$ is the norm of w vector.

The geometric margin is defined as
$\gamma_i=y_i\frac{(w^Tx_i+b)}{||w||}$

If $||w|| = 1$, then the geometric margin will be equal to the functional margin. The geometric margin is scale invariant; it does not change if we change w to 2w or b to 2b.

![Figure 2: Illustration of the geometric margin for a data point.](https://i.sstatic.net/wkM2a.png)

## Optimal margin classifier:

Now the goal of our SVM is to find a decision boundary that maximizes our geometric margin, as it would show a confident set of predictions and also correctly classify each data point.
So our maximum optimization problem can be represented by:

$$
\max_{\gamma,w,b} \quad \gamma
$$
subject to:
$$
y_i(w^Tx_i+b) \ge \gamma, \quad i=1, \dots, n
$$
$$
||w||=1
$$


The $||w||=1$ constraint makes sure that the functional margin is equal to the geometric margin. This optimization problem is a problematic one, as $||w||=1$ is a non-convex optimization, so to solve this, we will add a scaling constraint, as we have already discussed that introducing this scaling constraint does not change anything. We will put $\hat{\gamma}=1$ (the functional margin), so our optimization problem can be reduced to:

$$
\min_{w,b} \quad \frac{1}{2}||w||^2
$$
subject to: $$ y_i(w^Tx_i+b)\ge 1, \quad i=1, \dots, n $$


The solution to the above problem gives us an optimal margin classifier. Let’s look at the Lagrange dual form to further simplify this problem.
The Lagrange dual method has many advantages in this problem set, as it will lead to:
* Easier constraints
* Enable use of kernel for non-linear SVM
* Helps to identify support vectors

## Lagrange duality:

Our primal optimization problem for this will be:

$$ \min_{w,b} \quad \frac{1}{2}||w||^2 $$

subject to:

$$ y_i(w^Tx_i+b)\ge 1, \quad i=1, \dots, n $$


We can write the constraint as:

$$  g_i(w)=-y_i(w^Tx_i+b)+1 \le 0  $$

So, our Lagrangian for our problem will be:

$$ L(w,b,\alpha) = \frac{1}{2}||w||^2 - \sum_{i=1}^{n}\alpha_i[y_i(w^Tx_i+b)-1] $$


For our dual part of the problem, we will be minimizing with respect to $w$ and $b$ by differentiating with respect to $w$ and $b$.

$$ \nabla_wL(w,b,\alpha)=w-\sum_{i=1}^{n}\alpha_iy_ix_i $$


Putting it to zero, we will get:

$$ w=\sum_{i=1}^{n}\alpha_iy_ix_i $$


Now differentiating with respect to $b$:

$$ \frac{\partial L(w,b,\alpha)}{\partial b}=\sum_{i=1}^{n}\alpha_iy_i=0 $$


Putting these two results in the Lagrangian equation, we will get:

$$  W(\alpha) = \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i,j=1}^{n}y_iy_j\alpha_i\alpha_j \langle x_i^Tx_j \rangle  $$

where $$ \langle x_i^Tx_j \rangle $$ represents the inner product of our data points.

Now, our solution to this equation should satisfy the **Karush-Kuhn-Tucker (KKT) conditions**.
These conditions are:

1. Stationary:
   
$$[ \frac{\partial L(w,b,\alpha)}{\partial w}=0 ] $$

$$[\frac{\partial L(w,b,\alpha)}{\partial b}=0   ]$$

2. Dual feasibility:

$$[\alpha_i \ge 0]$$

3. Complementary slackness:

$$[\alpha_i(y_i(w^Tx_i+b)-1)=0]$$

   

From dual feasibility, if $\alpha_i > 0$ then $y_i(w^Tx_i+b)=1$. So our data points satisfying this condition will lie on the margin, as shown in the figure below, and these points are called **support vectors**.

![Figure 3: Illustration of the decision boundary, margin, and support vectors.](https://www.iunera.com/wp-content/uploads/image-16-1024x756.png?v=1596602837)

If $y_i(w^Tx_i+b) > 1$ then $\alpha_i=0$. Then these points will not influence our solutions.

4.  Primal Feasibility

    $$[y_i(w^Tx_i+b) \ge 1]$$
   

So, finally, our constraint combining our solutions and the KKT conditions will be:

$$ \max_{\alpha} \quad W(\alpha) = \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i,j=1}^{n}y_iy_j\alpha_i\alpha_j \langle x_i^Tx_j \rangle $$


subject to:


$$ \alpha_i \ge 0, \quad i=1, \dots, n $$


$$ \sum_{i=1}^{n}\alpha_iy_i=0 $$

## Conclusion:
In this post we saw how the maths behind SVM works.We began with intuitive goal of finding a decision boundary that maximize the margin between classes.This lead to a constraained optimization problem minimizing ||w|| 

To solve this we used **Lagrange dual problem** while satisfying Karush-Kuhn-Tucker (KKT) conditions. The dual formulation not only provides an elegant solution but also paves the way for one of the most powerful aspects of SVMs **the kernel trick**. By replacing the inner product  $$[\langle x_i^Tx_j \rangle]$$ with a kernel function, SVMs can efficiently create non-linear decision boundaries, making them a remarkably versatile and robust algorithm for complex classification tasks. 
