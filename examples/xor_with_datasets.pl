# Example demonstrating XOR

use strict;
use AI::NNFlex;
use AI::NNFlex::Dataset;

# Create the network 

my $network = AI::NNFlex->new([{"nodes"=>2,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation function"=>"tanh","random weights"=>1},
                        {"nodes"=>2,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation function"=>"tanh","random weights"=>1},
                       {"nodes"=>1,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation function"=>"linear","random weights"=>1}],
{'random connections'=>0,'networktype'=>'feedforward', 'random weights'=>1,'learning algorithm'=>'backprop','learning rate'=>.3,'debug'=>[],'bias'=>1});


my $dataset = AI::NNFlex::Dataset->new([
			[0,0],[0],
			[0,1],[1],
			[1,0],[1],
			[1,1],[0]]);


my $err = 10;
while ($err >.01)
{
	$err = $dataset->learn($network);

	# Alternative incorrect but supported syntax
	# (caught and translated by module)
	# $err = $network->learn($dataset);

	print "Error = $err\n";
}


# alternative incorrect but supported syntax
# foreach (@{$network->run($dataset)})
foreach (@{$dataset->run($network)})
{
	foreach (@$_){print $_}
	print "\n";	
}



