# Example demonstrating XOR with momentum backprop learning

use strict;
use AI::NNFlex;
use AI::NNFlex::Dataset;

# Create the network 

my $network = AI::NNFlex->new(randomconnections=>0,
				networktype=>'feedforward',
				randomweights=>1,
				learningalgorithm=>'momentum',
				learningrate=>.1,
				debug=>[],bias=>1,
				momentum=>0.6);



$network->add_layer(	nodes=>2,
			persistentactivation=>0,
			decay=>0.0,
			randomactivation=>0,
			threshold=>0.0,
			activationfunction=>"tanh",
			randomweights=>1);


$network->add_layer(	nodes=>2,
			persistentactivation=>0,
			decay=>0.0,
			randomactivation=>0,
			threshold=>0.0,
			activationfunction=>"tanh",
			randomweights=>1);

$network->add_layer(	nodes=>1,
			persistentactivation=>0,
			decay=>0.0,
			randomactivation=>0,
			threshold=>0.0,
			activationfunction=>"linear",
			randomweights=>1);


$network->init();

my $dataset = AI::NNFlex::Dataset->new([
			[0,0],[0],
			[0,1],[1],
			[1,0],[1],
			[1,1],[0]]);

my $counter=0;
my $err = 10;
while ($err >.01)
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



