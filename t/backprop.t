use strict;
use Test;
use AI::NNFlex::backprop;
use AI::NNFlex::Dataset;

BEGIN{
	plan tests=>7}

# test create network
my $network = AI::NNFlex::backprop->new(randomconnections=>0,
				randomweights=>1,
				learningrate=>.1,
				debug=>[],bias=>1,
				momentum=>0.6);

ok($network); #test 1
##

# test add layer
my $result = $network->add_layer(	nodes=>2,
			persistentactivation=>0,
			decay=>0.0,
			randomactivation=>0,
			threshold=>0.0,
			activationfunction=>"tanh",
			randomweights=>1);
ok($result); #test 2
##

# Test initialise network
$result = $network->init();
ok($result); #test 3
##

# test create dataset
my $dataset = AI::NNFlex::Dataset->new([
			[0,0],[1,1],
			[0,1],[1,0],
			[1,0],[0,1],
			[1,1],[0,0]]);
ok ($dataset); #test 4
##


# Test a learning pass
my $err = $dataset->learn($network);
ok($err); #test 5
##


# Test a run pass
$result = $dataset->run($network);
ok($result); #test 6
##

# test a lesion call
$result = $network->lesion(nodes=>0.5,connections=>0.5);
ok($result); #test 7
