AI::NNFlex

This module has developed quite a bit in the various versions, and backwards compatibility can be a little iffy. The current version is intended to constitute a base class for developing neural nets of all sorts, not just feedforward. I recast a lot of this when I started looking at developing a hopfield module (in progress). If you are seeking to create a neural network and don't really care about what goes on under the covers, you'll probably want the AI::NNFlex::Backprop module. Take a look at the xor.pl file in examples to see how to use it, if the pod doesn't make it clear enough.

From here on out, the interface should be pretty much consistent between versions - I'm concentrating now on making the pod easier to understand, developing a hopfield subclass, and developing a PDL subclass for very fast backprop.

Charles Colbourn
March 2005

###################################################################
Example XOR neural net (from examples/xor.pl


# Example demonstrating XOR with momentum backprop learning

use strict;
use AI::NNFlex::Backprop;
use AI::NNFlex::Dataset;

# Create the network 

my $network = AI::NNFlex::Backprop->new(
				learningrate=>.2,
				bias=>1,
				fahlmanconstant=>0.1,
				momentum=>0.6,
				round=>1);



$network->add_layer(	nodes=>2,
			activationfunction=>"tanh");


$network->add_layer(	nodes=>2,
			activationfunction=>"tanh");

$network->add_layer(	nodes=>1,
			activationfunction=>"linear");


$network->init();

my $dataset = AI::NNFlex::Dataset->new([
			[0,0],[0],
			[0,1],[1],
			[1,0],[1],
			[1,1],[0]]);



my $counter=0;
my $err = 10;
while ($err >.001)
{
	$err = $dataset->learn($network);
	print "Epoch = $counter error = $err\n";
	$counter++;
}


foreach (@{$dataset->run($network)})
{
	foreach (@$_){print $_}
	print "\n";	
}



