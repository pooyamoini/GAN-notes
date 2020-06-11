
# Generative Adversial Networks





## Maximum of objective function relative to the discriminator
We assume that the capacity of our model is unlimited and we wanna find the discriminator in which the objective function be its maximum. So we just need to the derivative of the objective relative to the discriminator be zero.

<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/978a4c296758645324cc9bc1ced4a1d8.svg?invert_in_darkmode" align=middle width=700.27427415pt height=95.99472300000001pt/>

Now we want to put the optimal discriminator into the objective function and see what's gonna happen.

<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/fed2f0a74b8d139337a342f1742261f4.svg?invert_in_darkmode" align=middle width=515.0328925499999pt height=24.65753399999998pt/>
<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/ae869795b87cc0dbc3d79086fd8d06fd.svg?invert_in_darkmode" align=middle width=700.2741042pt height=59.56936920000001pt/>
<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/4d50139d0a42903edf3261b5ab362e01.svg?invert_in_darkmode" align=middle width=459.32980335pt height=26.48417309999999pt/>
<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/819ead52cf36a1846f67f37006c8bba1.svg?invert_in_darkmode" align=middle width=667.3974796499999pt height=28.913154600000016pt/>

As we concluded , optimizing <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/472676a0c81bff80c31d97b32e1dc536.svg?invert_in_darkmode" align=middle width=58.95437129999999pt height=24.65753399999998pt/> where <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/f7b6d509278a223a72b23e097bcb9af1.svg?invert_in_darkmode" align=middle width=56.785287899999986pt height=22.63846199999998pt/> is equivalent to minimize the Jensen–Shannon distance of <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/f598c401c33319cba19a93076c3dbf0c.svg?invert_in_darkmode" align=middle width=34.34020094999999pt height=14.15524440000002pt/> and <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/7a999d5066af69a6a5ee75a93d7316b9.svg?invert_in_darkmode" align=middle width=18.50506019999999pt height=14.15524440000002pt/>. But all these results is in theory, in practice we can't assume that the capasity is unlimited. So in practice during a batch we update parameters of both discriminator and generative multiple times.

## Gradient problem

Assume that the dataset domain and generative domain do not overlap. Also assume that <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode" align=middle width=14.06623184999999pt height=22.465723500000017pt/> is so near to <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/2491ab92ce0d1ad314b21c144af61c94.svg?invert_in_darkmode" align=middle width=20.801426249999988pt height=22.63846199999998pt/>. 

We are going to calculate gradient of <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/46579f5c1bf40739904fad37c774bf03.svg?invert_in_darkmode" align=middle width=98.57539724999998pt height=24.65753399999998pt/> relative to network's logit. Discriminator is equivalent to <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/7a2b9254728e1d8ca5cc5d0ddee294ac.svg?invert_in_darkmode" align=middle width=31.45746944999999pt height=24.65753399999998pt/> which <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/a0f5bed1d2cfdb55f67ab895d66b8415.svg?invert_in_darkmode" align=middle width=62.60461844999999pt height=24.65753399999998pt/>. In this situation <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.68915409999999pt height=14.15524440000002pt/> is the logit.

<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/38092d1eaeded9340404de5f99e3514c.svg?invert_in_darkmode" align=middle width=592.46550825pt height=33.20539859999999pt/>

<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/b2da3971aa88b3ce2d2cacf607d84326.svg?invert_in_darkmode" align=middle width=340.3508361pt height=33.20539859999999pt/>

So whats the problem? <br/> 
If discriminator is near to the optimized, it will return 0 for the generative samples. So we have <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/fef1ea9d31adbe099f74baef2363df40.svg?invert_in_darkmode" align=middle width=119.75859224999998pt height=24.65753399999998pt/>. Therefore In accordance with the results we concluded, the gradient will be zero too. So the gradient won't be reached to the previous layers.<br/>
As the solution we suggest that use <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/2a082d88cc906779869b10ef52fded47.svg?invert_in_darkmode" align=middle width=75.74431424999999pt height=24.65753399999998pt/> as the loss function.

<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/7ad02bba2a9a0058430239a1a506fc17.svg?invert_in_darkmode" align=middle width=383.76217605pt height=27.77565449999998pt/> <br/>
<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/5f32d1809fcdb6e9ae7a08d859d3e796.svg?invert_in_darkmode" align=middle width=247.4145135pt height=33.20539859999999pt/>

So in this state, the gradient will be <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/0735e50df3d823678b38dd3d6a697ddf.svg?invert_in_darkmode" align=middle width=21.0046419pt height=21.18721440000001pt/> and the problem is solved.

## MLE-GAN

In this section, we're gonna analyze another approach we be equivalent to MLE approach. <br/>
In MLE approach, the loss function for generative is:
<p align="center"><img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/fb02895292b20973757f2aa92240bb6a.svg?invert_in_darkmode" align=middle width=238.79536394999997pt height=17.031940199999998pt/></p>
which <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> is the parameters of the generative network.<br/>
In our new approach the loss function for the discriminator wont change but the loss function for the generative is :
<p align="center"><img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/23a8cdd2c3ca58ab89a36a47951ffbe5.svg?invert_in_darkmode" align=middle width=215.5330485pt height=17.031940199999998pt/></p>
Now we want to calculate the function <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/843e4c9fa3daf579695c81ff8d23b263.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=24.65753399999998pt/>. <br/>
We have to find the fuction <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/843e4c9fa3daf579695c81ff8d23b263.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=24.65753399999998pt/> which gradients of <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/09d89095da71ee1df11c2268446cecb7.svg?invert_in_darkmode" align=middle width=86.44943999999998pt height=22.465723500000017pt/> respect to <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> be equal to gradients of <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/3e7846899c20a6dd6729abc0dede6534.svg?invert_in_darkmode" align=middle width=44.40885404999999pt height=22.465723500000017pt/> respect to <img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/>.<br/><br/>
<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/d602df2342d6c4c7edfaeec6b0e4bf14.svg?invert_in_darkmode" align=middle width=744.6349349999999pt height=31.56167850000001pt/> <br/>
<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/b0e619e49575d32807b845f894edadbb.svg?invert_in_darkmode" align=middle width=546.49680525pt height=33.20539859999999pt/> <br/>
<p align="center"><img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/fa96144fce24dc4bee8e05e2647ec45a.svg?invert_in_darkmode" align=middle width=134.88585989999999pt height=38.83491479999999pt/></p>
<img src="https://rawgit.com/pooyamoini/GAN-notes/master/svgs/83cf24cb2d035683491da94bd5de60d3.svg?invert_in_darkmode" align=middle width=343.3338909pt height=33.20539859999999pt/>


```
}
```
