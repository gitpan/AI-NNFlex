AI::NNFlex

This module has developed quite a bit in the various versions, and backwards compatibility can be a little iffy. The current version is intended to constitute a base class for developing neural nets of all sorts, not just feedforward. I recast a lot of this when I started looking at developing a hopfield module (in progress). If you are seeking to create a neural network and don't really care about what goes on under the covers, you'll probably want the AI::NNFlex::Backprop module. Take a look at the xor.pl file in examples to see how to use it, if the pod doesn't make it clear enough.

From here on out, the interface should be pretty much consistent between versions - I'm concentrating now on making the pod easier to understand, developing a hopfield subclass, and developing a PDL subclass for very fast backprop.

Charles Colbourn
March 2005

