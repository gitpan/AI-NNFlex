use strict;
use vars qw ($VERSION);
#use warnings;
###############################################################################
# NNFlex - Neural Network (flexible) - a heavily custom NN simulator
# 
# Sept 2004 - CW Colbourn
#
# This was developed from the abortive nnseq package originally intended
# for real time neural networks.
# The basis of the approach for this version is a very flexible, modular
# set of packages. This package constitutes the base, allowing the modeller
# to create meshes, apply input, and read output ONLY!
#
# Separate modules are to be written to perform feedback adjustments,
# various activation functions, text/gui front ends etc
#
###############################################################################
# Version Control
# ===============
#
# 0.1 20040905		CColbourn	New module
#					added NNFlex::datasets
#
# 0.11 20050113		CColbourn	Added NNFlex::lesion
#					Improved Draw
#					added NNFlex::datasets
#
# 0.12 20050116		CColbourn	Fixed reinforce.pm bug
# 					Added call into datasets
#					in ::run to offer alternative
#					syntax
#
# 0.13 20050121		CColbourn	Created momentum learning module
#
# 0.14 20050201		CColbourn	Abstracted derivatiive of activation
#					function into a separate function call
#					instead of hardcoded 1-y*y in backprop
#					tanh, linear & momentum
#
# 0.15 20050206		CColbourn	Fixed a bug in feedforward.pm. Stopped
#					calling dbug unless scalar debug > 0
#					in a lot of calls
#
###############################################################################
# ToDo
# ====
#
# Modify init to allow recurrent layer/node connections
# De-document node summing for now, as its mathematically equivalent
#  to node recurrent connections
# make the dump & load files a bit more intuitive (move the west? node across)
# write cmd & gui frontends
# simplifying the input from [{}{}{}]{} to {}{}{}x=>y,z=>1 should allow
#  the debugging to be simplified and the glabl variable to be removed
# Speed the bugger up!
# Revisit the node activation function method code - see if an object
#  method can be created neatly
#
# Odd thought - careful coding of a network would allow grafting of
# two different network types or learning algorithms, like an effectve
# single network with 2 layers unsupervised and 2 layers supervised
#
# Clean up the perldocs
#
###############################################################################
$VERSION = "0.15";


###############################################################################
my @DEBUG; 	# a single, solitary, shameful global variable. Couldn't
		#avoid it really. It allows correct control of debug
		#information before the $network object is created
		# (in ::layer->new & ::node->new for  example).


###############################################################################
###############################################################################
# package NNFlex
###############################################################################
###############################################################################
package AI::NNFlex;


=pod

=head1 NAME

AI::NNFlex - A customisable neural network simulator

=head1 SYNOPSIS

 use AI::NNFlex;

 my $network = AI::NNFlex->new(config parameter=>value);

 $network->add_layer(nodes=>x,activationfunction=>'function');

 $network->init(); 



 use AI::NNFlex::Dataset;

 my $dataset = AI::NNFlex::Dataset->new([
			[INPUTARRAY],[TARGETOUTPUT],
			[INPUTARRAY],[TARGETOUTPUT]]);

 my $sqrError = 10;

 while ($sqrError >0.01)

 {

	$sqrError = $dataset->learn($network);

 }

 $network->lesion({'nodes'=>PROBABILITY,'connections'=>PROBABILITY});

 my $outputsRef = $dataset->run($network);


=head1 DESCRIPTION

 AI::NNFlex is intended to be a highly flexible, modular NN framework.
 It's written entirely in native perl, so there are essentially no
 prereq's. The following modular divisions are made:

	* NNFlex.pm
		the core module. Contains methods to construct and
		lesion the network

	* feedforward.pm
		the network type module. Feedforward is the only type
		currently defined, but others may be created and
		imported at runtime

	* <learning>.pm
		the learning algorithm. Currently the options are
			backprop - standard vanilla backprop
			momentum - backprop with momentum


	* <activation>.pm
		node activation function. Currently the options are
		tanh, linear & sigmoid.

	* Dataset.pm
		methods for constructing a set of input/output data
		and applying to a network.

 The code should be simple enough to use for teaching
 purposes, but a simpler implementation of a simple backprop
 network is included in the example file bp.pl. This is
 derived from Phil Brierleys freely available java code
 at www.philbrierley.com.

 AI::NNFlex leans towards teaching NN and cognitive modelling
 applications. Future modules are likely to include more
 biologically plausible nets like DeVries & Principes
 Gamma model.

=head1 CONSTRUCTOR 

=head2 AI::NNFlex

 new ( parameter => value );
	
	randomweights=>MAXIMUM VALUE FOR INITIAL WEIGHT

	learningalgorithm=>The AI::NNFlex module to import for
		training the net

	networktype=>The AI::NNFlex module to import for flowing
		activation

	debug=>[LIST OF CODES FOR MODULES TO DEBUG]

	learningrate=>the learning rate of the network

	momentum=>the momentum value (momentum learning only)

=head2 AI::NNFlex::Dataset

 new ( [[INPUT VALUES],[OUTPUT VALUES],[INPUT VALUES],[OUTPUT VALUES],..])

=head3 INPUT VALUES

 These should be comma separated values. They can be applied
 to the network with ::run or ::learn

=head3 OUTPUT VALUES
	
 These are the intended or target output values. Comma separated
 These will be used by ::learn


=head1 METHODS

 This is a short list of the main methods. For details on all
 available methods, please see individual pod pages below, and
 in individual imported modules.

=head2 AI::NNFlex

=head3 add_layer

 Syntax:

 $network->add_layer(	nodes=>NUMBER OF NODES IN LAYER,
			persistentactivation=>RETAIN ACTIVATION BETWEEN PASSES,
			decay=>RATE OF ACTIVATION DECAY PER PASS,
			randomactivation=>MAXIMUM STARTING ACTIVATION,
			threshold=>NYI,
			activationfunction=>"ACTIVATION FUNCTION",
			randomweights=>MAX VALUE OF STARTING WEIGHTS);

=head3 init

 Syntax:

 $network->init();

 Initialises connections between nodes, sets initial weights and
 loads external components

=head3 lesion

 $network->lesion ({'nodes'=>PROBABILITY,'connections'=>PROBABILITY})

 Damages the network.

 PROBABILITY

 A value between 0 and 1, denoting the probability of a given node
 or connection being damaged.

 Note: this method may be called on a per network, per node or per
 layer basis using the appropriate object.

=head2 AN::NNFlex::Dataset

=head3 learn

 $dataset->learn($network)

 'Teaches' the network the dataset using the networks defined learning
 algorithm.
 Returns sqrError;

=head3 run

 $dataset->run($network)

 Runs the dataset through the network and returns a reference to an array of
 output patterns.

=head1 EXAMPLES

 See the code in ./examples. For any given version of NNFlex, xor.pl will
 contain the latest functionality.


=head1 PREREQs

 None. NNFlex should run OK on any version of Perl 5 >. 


=head1 ACKNOWLEDGEMENTS

 Phil Brierley, for his excellent free java code, that solved my backprop
 problem

 Dr Martin Le Voi, for help with concepts of NN in the early stages

 Dr David Plaut, for help with the project that this code was originally
 intended for.

 Graciliano M.Passos for suggestions & improved code (see SEE ALSO).

=head1 SEE ALSO

 AI::NNEasy - Developed by Graciliano M.Passos 
 Shares some common code with NNFlex. Much faster, and more suitable for
 backprop projects with large datasets.

=head1 TODO

 Lots of things:

 clean up the perldocs some more
 write gamma modules
 write BPTT modules
 write a perceptron learning module
 speed it up
 write a tk gui

=head1 CHANGES

 v0.11 introduces the lesion method, png support in the draw module
  and datasets.
 v0.12 fixes a bug in reinforce.pm & adds a reflector in feedforward->run
  to make $network->run($dataset) work.
 v0.13 introduces the momentum learning algorithm and fixes a bug that
  allowed training to proceed even if the node activation function module
  can't be loaded
 v0.14 fixes momentum and backprop so they are no longer nailed to tanh hidden
  units only.
 v0.15 fixes a bug in feedforward, and reduces the debug overhead
  

=head1 COPYRIGHT

 Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program
 is free software; you can redistribute it and/or modify it under the same
 terms as Perl itself.


=head1 CONTACT

 charlesc@nnflex.g0n.net



=cut

###############################################################################




###############################################################################
# AI::NNFlex::new
###############################################################################

#=pod
#
#Below are PODs for individual methods
#
#
#=head1 AI::NNFlex::new
#
#=head2 constructor - creates new neural network
#
#takes as params a reference to a hash. Each hash will be created as a layer.
#
#Valid parameters are currently:
#
#	* nodes			-	number of nodes in the layer
#
#	* decay			-	float, amount that each node in the
#					layer will decay with each tick.
#
#	* persistentactivation -	whether activation is summed between
#					ticks.
#
#	* adjusterror		-	<NYI>
#
#	* randomactivation	-	parameter to pass to RAND for random
#					activation seeding
#
#	Additional parameters may be specified as individual param=>value pairs
#	AFTER the layer hash. These may include:
#
#	* randomweights	-	parameter to pass to RAND for random
#					weights seeding
#	
#	* randomconnections	-	The /probability/ factor of a connection
#					being made between two nodes in the
#					network.
#
#					(Note that no support exists at present
#					for restricting the connectivity at a
#					layer by layer level, although this may
#					be done by combining two or more network
#					objects and handling activation from one
#					to the next programmatically)
#
#	* learningalgorithm	-	the learning algorithm (this must be a
#					valid compatible perl module).
#
#	* networktype		-	E.g. feedforward. Must be a compatible
#					perl module.
#
#	* debug			-	level of debug information
#					a different debug level is assigned to
#					each module type, and the debug property is
#					an ARRAY of which debugs you require.
#					0 - error
#					1 - TBD
#					2 - NNFlex.pm core debug
#					3 - networktype debug
#					4 - learning algorithm debug
#					5 - activation function debug
#					6 - GUI/Graphic
#
#Plus other custom settings used in networktype & learning algorithm modules, such as:
#
#	* learning rate		-	A constant for use in e.g. backprop learning
#
#
#
#Returns a network object that contains $$network->{'layers'}
#which is an array of 'layer' objects. 
#
#The layer object contains a property 'nodes' which is an array of nodes
#in that layer. So programmatically if you want to access a particular
#node (or to interact with the mesh for writing networktypes and learning
#algorithms) you can access any node directly using the syntax
#
#$network->{'layers'}->[layer number]->{'nodes'}->[node number]->{property}
#
#(HINT: or do foreach's on the arrays)
#
#Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify
#it under the same terms as Perl itself.
#
##=cut

###############################################################################
sub new
{
	my $class = shift;
	my $network={};
	bless $network,$class;

	# intercept the new style 'empty network' constructor call
	# Maybe I should deprecate the old one, but its convenient, provided you
	# can follow the mess of hashes
	
	if (!grep /HASH/,@_)
	{
		my %config = @_;
		foreach (keys %config)
		{
			$network->{$_} = $config{$_};
		}

		return $network;
	}

	# Otherwise, continue assuming that the whole network is defined in 
	# a pair of anonymous hashes	



	my $params = shift;
	my $netParams = shift;
	my @layers;
	dbug ($netParams,"Entered AI::NNFlex::new with params $params $netParams",2);


	# clean up case & spaces in layer defs from pre 0.14 constructor calls:
	my $cleanParams;
	foreach my $layer(@{$params})
	{
		my %cleanLayer;
		foreach (keys %$layer)
		{
			my $key = lc($_);
			$key =~ s/\s//g;
			$cleanLayer{$key} = $$layer{$_};
		}
		push @$cleanParams,\%cleanLayer;
	}



	# Network wide parameters (e.g. random weights)
	foreach (keys %$netParams)
	{
		my $key = lc($_);
		$key =~ s/\s//g; # up to 0.14 we had params with spaces in, now deprecated
		$network->{$key} = ${$netParams}{$_};
	}

	if( $network->{'debug'})
	{
		@DEBUG = @{$network->{'debug'}};
	}

	# build the network
	foreach (@$cleanParams)
	{
		if (!($$_{'nodes'})){next}
		my %layer = %{$_};	
		push @layers,AI::NNFlex::layer->new(\%layer);	
	}	
	$$network{'layers'} = \@layers;





	$network->init;
	return $network;
}




###############################################################################
# AI::NNFlex::add_layer
###############################################################################
#
# Adds a layer of given node definitions to the $network object
#
# syntax
#
# $network->add_layer(nodes=>4,activationfunction=>tanh);
#
# returns bool success or failure
#
###############################################################################

sub add_layer
{
	my $network = shift;

	my %config = @_;

	my $layer = AI::NNFlex::layer->new(\%config);

	if ($layer)
	{
		push @{$network->{'layers'}},$layer;
		return 1;
	}
	else
	{
		return 0;
	}
}


###############################################################################
# AI::NNFlex::output
###############################################################################
#=pod 
#
#=head1 AI::NNFlex::output
#
#
#$object->output({"output"=>"1"}); returns the activation of layer 1
#
#
#else returns activation of last layer as a reference to an array
#
##=cut

###############################################################################
sub output
{
	my $network = shift;
	my $params = shift;

	my $finalLayer = ${$$network{'layers'}}[-1];

	my $outputLayer;

	if (defined $$params{'layer'})
	{
		$outputLayer = ${$$network{'layers'}}[$$params{'layer'}]
	}
	else
	{
		$outputLayer = $finalLayer
	}

	my $output = AI::NNFlex::layer::layer_output($outputLayer);
	return $output;
}
################################################################################
# sub init
################################################################################

#=pod
#
#=head1 AI::NNFlex::init
#
#
#
#called from AI::NNFlex::new. no external use required, but not defined as local, in case of debugging use
#
#Init runs through each layer of node objects, creating properties in each node:
#
#	* connectedNodesEast	-	Nodes connected to this node in the layer
#					to the 'east', i.e. next layer in feedforward
#					terms.
#
#	* connectedNodesWest	-	Nodes connected to this node in the layer to
#					the west, i.e. previous layer in feedforward
#					networks.
#
#These properties are hashes, with the node object acting as a key. Each value
#is a weight for this connection. This means that you /can/ have connection weights
#for connections in both directions, since the weight is associated with an incoming
#connection.
#
#access with the following syntax:
#
#$node->{'connectedNodesWest'}->{'weights'}->{$connectedNode} = 0.12345
#$node->{'connectedNodesWest'}->{'nodes'}->[number] = $nodeObject
#
#
#Note also that the probability of a connection being created is equal to the numeric 
#value of the global property 'random connections' expressed as a decimal between 0 and 1.
#If 'random connections' is not specified all connections will be made.
#
#The connections are /only/ created from west to east. Connections that already exist from
#west to east are just copied for the 'connectedNodesWest' property.
#
#No return value: the connections are created in the $network object.
#
#
#These connectedNodesEast & West are handy, because they are arrays you can foreach
#them to iterate through the connected nodes in a layer to the east or west.
#
##=cut

###############################################################################
sub init
{

	#Revised version of init for NNFlex

	my $network = shift;
	my @layers = @{$network->{'layers'}};

	my @debug = @{$network->{'debug'}};

	# one of the parameters will normally be networktype
	# this specifies a class for activation flow, which
	# is included here
	if( $network->{'networktype'})
	{
		my $requirestring = "require \"AI/NNFlex/".$network->{'networktype'}.".pm\"";
		if (!eval($requirestring)){die "Can't load type ".$network->{'networktype'}.".pm because $@\n"};
	}
	if( $network->{'learningalgorithm'})
	{
		my $requirestring = "require \"AI/NNFlex/".$network->{'learningalgorithm'}.".pm\"";
		if (!eval($requirestring)){die "Can't load ".$network->{'learningalgorithm'}.".pm because $@\n"};
	}

	# implement the bias node
	if ($network->{'bias'})
	{
		my $biasNode = AI::NNFlex::node->new({'activation function'=>'linear'});
		$$network{'biasNode'} = $biasNode;
		$$network{'biasNode'}->{'activation'} = 1;
	}

	my $nodeid = 1;
	my $currentLayer=0;	
	# foreach layer, we need to examine each node
	foreach my $layer (@layers)
	{
		# Foreach node we need to make connections east and west
		foreach my $node (@{$layer->{'nodes'}})
		{
			# import the nodes activation function
			my $requireString = "require (\"AI/NNFlex/".$node->{'activationfunction'}.".pm\")";
			if (!eval ($requireString)){die "Can't load activation function ".$node->{'activationfunction'}.".pm\n"};
			$node->{'nodeid'} = $nodeid;
			# only initialise to the west if layer > 0
			if ($currentLayer > 0 )
			{
				foreach my $westNodes (@{$network->{'layers'}->[$currentLayer -1]->{'nodes'}})	
				{
					foreach my $connectionFromWest (@{$westNodes->{'connectedNodesEast'}->{'nodes'}})
					{
						if ($connectionFromWest eq $node)
						{
							my $weight;
							if ($network->{'randomweights'})
							{
								$weight = rand(2)-1;
							}
							else
							{
								$weight = 0;
							}
							push @{$node->{'connectedNodesWest'}->{'nodes'}},$westNodes;
							${$node->{'connectedNodesWest'}->{'weights'}}{$westNodes} = $weight;
							if (scalar @debug > 0)	
							{$network->dbug ("West to east Connection - $westNodes to $node",2);}
						}
					}
				}
			}

			# Now initialise connections to the east (if not last layer)
			if ($currentLayer < (scalar @layers)-1)
			{
			foreach my $eastNodes (@{$network->{'layers'}->[$currentLayer+1]->{'nodes'}})
			{
				if (!$network->{'randomconnections'}  || $network->{'randomconnections'} > rand(1))
				{
					my $weight;
					if ($network->{'randomweights'})
					{
						$weight = rand(1);
					}
					else
					{
						$weight = 0;
					}
					push @{$node->{'connectedNodesEast'}->{'nodes'}},$eastNodes;
					${$node->{'connectedNodesEast'}->{'weights'}}{$eastNodes} = $weight;
					if (scalar @debug > 0)
					{$network->dbug ("East to west Connection $node to $eastNodes",2);}
				}
			}
			}
			$nodeid++;
		}
		$currentLayer++;
	}


	# add bias node to westerly connections
	if ($network->{'bias'})
	{
		foreach my $layer (@{$network->{'layers'}})
		{
			foreach my $node (@{$layer->{'nodes'}})
			{
				push @{$node->{'connectedNodesWest'}->{'nodes'}},$network->{'biasNode'};
				my $weight;
				if ($network->{'randomweights'})
				{
					$weight = rand(1);
				}
				else
				{
					$weight = 0;
				}
				${$node->{'connectedNodesWest'}->{'weights'}}{$network->{'biasNode'}} = $weight;
				if (scalar @debug > 0)
				{$network->dbug ("West to east Connection - ".$network->{'biasNode'}." to $node weight = $weight",2);}
			}
		}
	}



	return 1; # return success if we get to here


}

###############################################################################
# sub $network->dbug
###############################################################################

#=pod
#
#=head1 AI::NNFlex::$network->dbug
#
#
#
#Internal use, writes to STDOUT parameter 1 if parameter 2 == global variable $DEBUG.
#or parameter 2 == 1
#
##=cut

###############################################################################
sub dbug
{
	my $network = shift;
	my $message = shift;
	my $level = shift;


	my @DEBUGLEVELS;
	# cover for debug calls before the network is created
	if (!$network->{'debug'})
	{
		@DEBUGLEVELS=@DEBUG;
	}
	else
	{
		@DEBUGLEVELS = @{$network->{'debug'}};
	}


	# 0 is error so ALWAYS display
	if (!(grep /0/,@DEBUGLEVELS)){push @DEBUGLEVELS,0}

	foreach (@DEBUGLEVELS)
	{
	
		if ($level == $_)
		{
			print "$message\n";
		}
	}
}


###############################################################################
# AI::NNFlex::dump_state
###############################################################################

#=pod
#
#=head1 AI::NNFlex::dump_state
#
#
#
#$network->dump_state({"filename"=>"test.wts"[, "activations"=>1]});
#
#Dumps the current contents of the node weights to a file.
#
##=cut

###############################################################################

sub dump_state
{
	my $network = shift;
	my $params =shift;
	# This needs rewriting. It should include all connection weights, e->w & w->e


	my $filename = $$params{'filename'};
	my $activations = $$params{'activations'};

	
	open (OFILE,">$filename") or return "Can't create weights file $filename";


	foreach my $layer (@{$network->{'layers'}})
	{
		foreach my $node (@{$layer->{'nodes'}})
		{
			if ($activations)
			{
				print OFILE $node->{'nodeid'}." activation = ".$node->{'activation'}."\n";
			}
			foreach my $connectedNode (@{$node->{'connectedNodesEast'}->{'nodes'}})
			{
				my $weight = ${$node->{'connectedNodesEast'}->{'weights'}}{$connectedNode};
				print OFILE $node->{'nodeid'}." <- ".$connectedNode->{'nodeid'}." = ".$weight."\n";
			}

			if ($node->{'connectedNodesWest'})
			{
				foreach my $connectedNode (@{$node->{'connectedNodesWest'}->{'nodes'}})
				{
					#FIXME - a more easily read format would be connectedNode first in the file
					my $weight = ${$node->{'connectedNodesWest'}->{'weights'}}{$connectedNode};
					print OFILE $node->{'nodeid'}." -> ".$connectedNode->{'nodeid'}." = ".$weight."\n";
				}
			}
		}
	}




	close OFILE;
}

###############################################################################
# sub load_state
###############################################################################
#=pod
#
#=head1 load_state
#
#useage:
#	$network->load_state(<filename>);
#
#Initialises the network with the state information (weights and, optionally
#activation) from the specified filename.
#
#Note that if you have a file containing activation, but you are not using
#persistent activation, the activation states of nodes will be reset during
#network->run
#
##=cut
################################################################################
sub load_state
{
	my $network = shift;
	my $filename = shift;

	open (IFILE,$filename) or return "Error: unable to open $filename because $!";

	# we have to build a map of nodeids to objects
	my %nodeMap;
	foreach my $layer (@{$network->{'layers'}})
	{
		foreach my $node (@{$layer->{'nodes'}})
		{
			$nodeMap{$node->{'nodeid'}} = $node;
		}
	}



	my %stateFromFile;

	while (<IFILE>)
	{
		chomp $_;
		my ($activation,$nodeid,$destNode,$weight);

		if ($_ =~ /(.*) activation = (.*)/)
		{
			$nodeid = $1;
			$activation = $2;
			$stateFromFile{$nodeid}->{'activation'} = $activation;
			$network->dbug("Loading $nodeid = $activation",2);
		}
		elsif ($_ =~ /(.*) -> (.*) = (.*)/)
		{
			$nodeid = $1;
			$destNode = $2;
			$weight = $3;
			$network->dbug("Loading $nodeid -> $destNode = $weight",2);
			$stateFromFile{$nodeid}->{'connectedNodesWest'}->{'weights'}->{$nodeMap{$destNode}}=$weight;
			push @{$stateFromFile{$nodeid}->{'connectedNodesWest'}->{'nodes'}},$nodeMap{$destNode};
		}	
		elsif ($_ =~ /(.*) <- (.*) = (.*)/)
		{
			$nodeid = $1;
			$destNode = $2;
			$weight = $3;
			$stateFromFile{$nodeid}->{'connectedNodesEast'}->{'weights'}->{$nodeMap{$destNode}}=$weight;
			push @{$stateFromFile{$nodeid}->{'connectedNodesEast'}->{'nodes'}},$nodeMap{$destNode};
			$network->dbug("Loading $nodeid <- $destNode = $weight",2);
		}	
	}

	close IFILE;




	my $nodeCounter=1;

	foreach my $layer (@{$network->{'layers'}})
	{
		foreach my $node (@{$layer->{'nodes'}})
		{
			$node->{'activation'} = $stateFromFile{$nodeCounter}->{'activation'};
			$node->{'connectedNodesEast'} = $stateFromFile{$nodeCounter}->{'connectedNodesEast'};
			$node->{'connectedNodesWest'} = $stateFromFile{$nodeCounter}->{'connectedNodesWest'};
			$nodeCounter++;
		}
	}


}

##############################################################################
# sub lesion
##############################################################################
#=pod
#
#=head1 AI::NNFlex::lesion
#
#=item
#
#Calls node::lesion for each node in each layer
#
#Lesions a node to emulate damage. Syntax is as follows
#
#$network->lesion({'nodes'=>.2,'connections'=>.4});
#
#assigns a .2 probability of a given node being lesioned, and
#.4 probability of a given connection being lesioned. Either option can
#be omitted but it must have one or the other to do. If you programmatically
#need to call it with no lesioning to be done, call with a 0 probability of
#lesioning for one of the options.
#
#return value is true if successful;
#
#
##=cut
##########################################################
sub lesion
{
	
        my $network = shift;

        my $params =  shift;
	my $return;
        $network->dbug("Entered AI::NNFlex::lesion with $params",2);

        my $nodeLesion = $$params{'nodes'};
        my $connectionLesion = $$params{'connections'};

        # go through the layers & node inactivating random nodes according
        # to probability
        
	foreach my $layer (@{$network->{'layers'}})
	{
		$return = $layer->lesion($params);
	}

	return $return;

}








###############################################################################
###############################################################################
# Package AI::NNFlex::layer
###############################################################################
###############################################################################
package AI::NNFlex::layer;

#=pod
#
#=head1 AI::NNFlex::layer
#
#
#
#The layer object
#
##=cut

###############################################################################
# AI::NNFlex::layer::new
###############################################################################

#=pod
#
#=head1 AI::NNFlex::layer::new
#
#
#
#Create new layer
#
#Takes the parameters from AI::NNFlex::layer and passes them through to AI::NNFlex::node::new
#
#(Uses nodes=>X to decide how many nodes to create)
#
#Returns a layer object containing an array of node objects
#
##=cut

###############################################################################
sub new
{
	my $layer={};
	my $class = shift;
	my $params = shift;
	bless $layer,$class;
	 
	my $numNodes = $$params{'nodes'};
	
	my @nodes;

	for (1..$numNodes)
	{
		push @nodes, AI::NNFlex::node->new($params);
	}

	$$layer{'nodes'} = \@nodes;
	AI::NNFlex::dbug($params,"Created layer $layer",2);
	return $layer;
}

###############################################################################
# AI::NNFlex::layer::layer_output
##############################################################################

#=pod
#
#=head1 AI::NNFlex::layer::layer_output
#
#
#
#Receives a reference to a hash of parameters. Valid inputs are
#
#	* layer		-	the layer number you want output from
#
#Returns a reference to an array of outputs
#
#Used by AI::NNFlex::output
#
#=cut

###############################################################################
sub layer_output
{
	my $layer = shift;
	my $params = shift;


	my @outputs;
	foreach my $node (@{$$layer{'nodes'}})
	{
		push @outputs,$$node{'activation'};
	}

	return \@outputs;	
}



##############################################################################
# sub lesion
##############################################################################
#=pod
#
#=head1 AI::NNFlex::layer::lesion
#
#=item
#
#Calls node::lesion for each node in the layer
#
#Lesions a node to emulate damage. Syntax is as follows
#
#$layer->lesion({'nodes'=>.2,'connections'=>.4});
#
#assigns a .2 probability of a given node being lesioned, and
#.4 probability of a given connection being lesioned. Either option can
#be omitted but it must have one or the other to do. If you programmatically
#need to call it with no lesioning to be done, call with a 0 probability of
#lesioning for one of the options.
#
#return value is true if successful;
#
#
#=cut
##########################################################
sub lesion
{
        
        my $layer = shift;

        my $params =  shift;
        my $return;


        my $nodeLesion = $$params{'nodes'};
        my $connectionLesion = $$params{'connections'};

        # go through the layers & node inactivating random nodes according
        # to probability
        
        foreach my $node (@{$layer->{'nodes'}})
        {
                $return = $node->lesion($params);
        }

        return $return;

}



###############################################################################
###############################################################################
# package AI::NNFlex::node
###############################################################################
###############################################################################
package AI::NNFlex::node;

#=pod
#
#=head1 AI::NNFlex::node
#
#
#
#the node object
#
#=cut

###############################################################################
# AI::NNFlex::node::new
###############################################################################

#=pod
#
#=head1 AI::NNFlex::node::new
#
#Takes parameters passed from NNFlex via AI::NNFlex::layer
#
#returns a node object containing:
#
#	* activation		-	the nodes current activation
#
#	* decay			-	decay rate
#
#	* adjust error		-	NYI
#
#	* persistent activation -	if true, activation will be summed on
#					each run, rather than zeroed
#					before calculating inputs.
#
#	* ID			-	node identifier (unique across the
#					NNFlex object)
#
#	* threshold		-	the level above which the node
#					becomes active
#
#	* activation function	-	the perl script used as an activation
#					function. Must perform the calculation
#					on a variable called $value.
#
#	* active		-	whether the node is active or
#					not. For lesioning. Set to 1
#					on creation
#
# NOTE: it is recommended that nothing programmatic be done with ID. This
# is intended to be used for human reference only.
#
#
#=cut

###############################################################################
sub new
{
	my $class = shift;
	my $node = {};
	my $params = shift;

	if ($$params{'randomactivation'})
	{
		$$node{'activation'} = 
			rand($$params{'random'});
			AI::NNFlex::dbug($params,"Randomly activated at ".$$node{'activation'},2);
	}
	else
	{
		$$node{'activation'} = 0;
	}
	$$node{'randomweights'} = $$params{'randomweights'};
	$$node{'decay'} = $$params{'decay'};
	$$node{'adjusterror'} = $$params{'adjusterror'};
	$$node{'persistentactivation'} = $$params{'persistentactivation'};
	$$node{'threshold'} = $$params{'threshold'};
	$$node{'activationfunction'} = $$params{'activationfunction'};
	$$node{'active'} = 1;
	

	$$node{'error'} = 0;
	
	bless $node,$class;	
	AI::NNFlex::dbug($params,"Created node $node",2);
	return $node;
}

##############################################################################
# sub lesion
##############################################################################
#=pod 
#
#=head1 AI::NNFlex::node::lesion
#
#=item
#
#Lesions a node to emulate damage. Syntax is as follows
#
#$node->lesion({'nodes'=>.2,'connections'=>.4});
#
#assigns a .2 probability of a given node being lesioned, and 
#.4 probability of a given connection being lesioned. Either option can
#be omitted but it must have one or the other to do. If you programmatically
#need to call it with no lesioning to be done, call with a 0 probability of
#lesioning for one of the options.
#
#return value is true if successful;
#
#Implemented as a method of node to permit node by node or layer by layer
#lesioning
#
#=cut

###############################################################################
sub lesion
{

        my $node = shift;

        my $params =  shift;


        my $nodeLesion = $$params{'nodes'};
        my $connectionLesion = $$params{'connections'};

        # go through the layers & node inactivating random nodes according
        # to probability
        
	if ($nodeLesion)
	{
		my $probability = rand(1);
		if ($probability < $nodeLesion)
		{
			$node->{'active'} = 0;
		}
	}

	if ($connectionLesion)
	{
		# init works from west to east, so we should here too
		my $nodeCounter=0;
		foreach my $connectedNode (@{$node->{'connectedNodesEast'}->{'nodes'}})
		{
			my $probability = rand(1);
			if ($probability < $connectionLesion)
			{
				my $reverseNodeCounter=0; # maybe should have done this differntly in init, but 2 late now!
				${$node->{'connectedNodesEast'}->{'nodes'}}[$nodeCounter] = undef;
				foreach my $reverseConnection (@{$connectedNode->{'connectedNodesWest'}->{'nodes'}})
				{
					if ($reverseConnection == $node)
					{
						${$connectedNode->{'connectedNodesEast'}->{'nodes'}}[$reverseNodeCounter] = undef;
					}
					$reverseNodeCounter++;
				}

			}
                                
			$nodeCounter++;
		}


        }
        

        return 1;
}

1;

