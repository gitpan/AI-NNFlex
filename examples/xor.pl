# Example demonstrating XOR with momentum backprop learning

use strict;
use AI::NNFlex::Backprop;
use AI::NNFlex::Dataset;

# Create the network 

my $network = AI::NNFlex::Backprop->new(
				learningrate=>.15,
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

	print "Epoch $counter: Error = $err\n";
	$counter++;
}


foreach (@{$dataset->run($network)})
{
	foreach (@$_){print $_}
	print "\n";	
}



