use strict;
use vars qw ($VERSION);
#use warnings;
###############################################################################
# nnflex - Neural Network (flexible) - a heavily custom NN simulator
# 
# Sept 2004 - CW Colbourn
#
# This was developed from the abortive nnseq package originally intended
# for real time neural networks.
# The basis of the approach for this version is a very flexible, modular
# set of packages. This package constitutes the base, allowing the modeller
# to create meshes, apply input, and read output ONLY!
# (oh, and maybe lesion - CWC)
#
# Separate modules are to be written to perform feedback adjustments,
# various activation functions, text/gui front ends etc
#
###############################################################################
# Version Control
# ===============
#
# 0.1 20040905		CColbourn	New module
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
# write lesioning method - node will probably need active=1/0 attribute
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
# an experimental training algorithm - strengthen any active connection only
# (and weaken inactive connections). 
#
# Clean up the perldocs
#
###############################################################################
$VERSION = "0.1";


###############################################################################
my @DEBUG; 	# a single, solitary, shameful global variable. Couldn't
		#avoid it really. It allows correct control of debug
		#information before the $network object is created
		# (in ::layer->new & ::node->new for  example).


###############################################################################
###############################################################################
# package nnflex
###############################################################################
###############################################################################
package AI::nnflex;


=pod
=head1 AI::nnflex - Neural Network (Flexible)

=head2 a heavily custom neural network simulator

=item

NNFlex - a flexible native perl API for neural networks

Author: Charles Colbourn
Version: 0.1
Date: November 2004

NNFLEX is intended to be a flexible framework for
developing neural network applications. The code is
moderately complex and not really suitable for learning
the basic functioning of the backpropagation algorithm.
Use Phil Brierleys excellent example code at
http://www.philbrierley.com
for that (it's in java and vb on the site, If you need
to read it in perl, there is a translated version in
this package, filename bp.pl).


Rationale:
Why did I do this? Well, first of all, why not. Perl is
my main programming language, and I was having trouble 
following the example code in other languages. I needed
a customisable NN framework for an MSc project, and had
been unable to get AI-neuralnet-mesh or
AI-neuralnet-Backprop working. So I thought I'd write my
own, as a learning experience.

Why perl? Because I like perl; because its the language
I write most easily; and because, as a linguist of sorts
its the best language for me to use. I know its much
slower than C++, but I don't really care. Xerion
deals with my big modelling very quickly - this does the
stuff that I need to fiddle with. I will concede that it
is /dog/ slow (WIP to improve that), but its quick to 
make changes to network architecture, without having to
figure out the deeper parts of the code.

Plus, as a big motivation, I wanted something easily
portable, pastable into an SSH session if I couldn't get
FTP access to a machine, and easily installable without
root access. Perl modules fulfill all these criteria.

Compatibility/Dependencies.
Perl. Basically thats it. I habitually code with 
strict refs on, but if you have one of these cut down
'shipped with the oracle client' types of perl installation
you can safely remove 'use strict' from the top of the 
modules and run it. I wrote some of it on perl 5.005_51 
for AIX, and it works fine on that. I use 5.8 on my laptop
where it also runs fine, so anything else should be OK.

There are no additional modules required, although in the
future I plan to implement a Tk GUI - but running that
will be optional anyway.


Usage:

The current version of nnflex is basically just an API,
rather than a modelling package. A more user friendly
frontend may follow later. To create a network, use the
following syntax:

use nnflex;
my $network = nnflex->new([array of hashes],{hash of global config});

for example:

my $object = nnflex->new([{"nodes"=>2,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation fu
nction"=>"tanh","random weights"=>1},
                        {"nodes"=>2,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation func
tion"=>"tanh","random weights"=>1},
                       {"nodes"=>1,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation funct
ion"=>"linear","random weights"=>1}],{'random connections'=>0,'networktype'=>'feedforward', 'random weights'=>1,'learning algorithm'
=>'backprop','learning rate'=>.3,'debug'=>[],'bias'=>1});

The array of hashes contains a single {} set for each
layer in the network. Global config options like
learning rates, debug, whether bias is used etc are
contained in a separate hash after the array of layers.

The object returned is an nnflex object, which contains
a number of properties and methods. This is where it gets
complicated, because exactly what properties and methods
depends on what has been defined in config. In particular
the following configuration options in global:

'networktype'
'learning algorithm'

do some special stuff. Whatever you specify for these
two options is /imported/ as a perl module. For example, 
if you specify the networktype to be 'feedforward', the
feedforward.pm module will be included. Calls to 
$network->run
will then call the feedforward implementation of run.

Likewise, if you specify
'learning algorithm'=>'backprop'
then the backpropagation algorithm module 'backprop.pm'
will be included, and any calls to 'learn' will use the
learn method defined in there.

The nnflex object has a property called 'layers'. This
is an array containing an AI::nnflex::layer object for each
layer in the network.

The AI::nnflex::layer object has a property called 'nodes'.
This is an array containing AI::nnflex::node objects, one
for each node in the layer.

The AI::nnflex::node object has a whole host of properties,
see the perldoc for full details. Among them are:
* activation    - the current activation of the node
* activation function
                - the perl module used to calculate
                  the nodes activation.
* id            - the numeric ID of the node
* error         - the current network error of the node
* connectedNodesWest
                - an array of all nodes connected to the
                  'west'
* connectedNodesEast
                - an array of all nodes connected to the
                  'east'

This lot could probably benefit from some explanation. 
First of all 'activation'. The network is 'stateful'. If
you run a stimulus set through it, it will remain at
the same level of activation until you reset it.

'Activation function' - this is a little perl module that
performs the activation function.

ConnectedNodes. I wanted to make this code as flexible
as possible, and particularly to leave potential to 
implement different kinds of network types and learning
algorithms, so each node knows not only what nodes it is
connected to 'upstream' in backprop terms, but also
'downstream'. This should allow development of bidirectional
networks like 'Hemholtz'(?). Be warned though - at present
the bias node (heavily used in backprop) is added to the
list of westerly nodes but has no easterly list.

The upshot of this is that you can access (and change)
anything at any time. For example, you can get the activation
of a node using:
$network->{'layers'}->[<layer number>]->{'nodes'}->[<node number within layer>]->{'activation'}


Basically the structure is intended to allow me (or anyone
else for that matter) to develop new NN code and network
types very quickly, without having to rewrite &| modify
whopping great chunks of code.

TIP: if you find that the out-of-the-box backprop code is
failing to converge, and error is in fact increasing, drop
your learning constant. I experimented to get 0.3 for XOR,
0.1 seems to work OK for a network of 48 nodes, (holds up
moistened finger) for each double after this I insert
another decimal point.


VERSION
0.1

ACKNOWLEDGEMENTS
Phil Brierley, for making his excellent example code
(which dug me out of a backprop hole) available to 
all and sundry.

Josiah Bryan, for writing the original native perl
backprop modules, which I've used for inspiration in
the way the API interface is constructed.

Dr Martin Le Voi, for considerable amounts of help when
trying to get my head around the fundamentals of backprop.

BUGS, FEATURE REQUESTS AND PATCHES
Please feel free to amend, improve and patch nnflex to 
whatever extent you wish. Any really nice additional
functionality, please mail it to me (as a diff, an 
amended module/script, or a description) at the address
below. Please don't distribute amended versions of the
code without my say so (see COPYRIGHT).

I'm happy to write additional functionality, time
permitting. Time is something I don't have a lot of, so
anything big, be prepared to wait up to a year, or write
it yourself.

If you do so, please try to stick to something like the
conventions set out in the code. In particular, learning
algorithms should be implemented as modules, as should
new network types. These should have 'run' and 'learn' 
methods respectively as a minimum. Where possible please
don't include anything outside of standard core perl,
and please leave strict ON. All this only applies to code
sent back to me to be included in the distribution of 
course, you can do what you like to your own copy - that
is the whole point!

PLANNED IMPROVEMENTS
more user friendly frontend.
improved speed!
Tk based GUI frontend
gamma network module (DHYB)


COPYRIGHT
Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify 
it under the same terms as Perl itself.


CONTACT

charlesc@nnflex.g0n.net



=cut

###############################################################################




###############################################################################
# AI::nnflex::new
###############################################################################

=pod

=head1 AI::nnflex::new

=head2 constructor - creates new neural network

takes as params a reference to a hash. Each hash will be created as a layer.

Valid parameters are currently:

	* nodes			-	number of nodes in the layer

	* decay			-	float, amount that each node in the
					layer will decay with each tick.

	* persistent activation -	whether activation is summed between
					ticks.

	* adjust error		-	<NYI>

	* random activation	-	parameter to pass to RAND for random
					activation seeding

	Additional parameters may be specified as individual param=>value pairs
	AFTER the layer hash. These may include:

	* random weights	-	parameter to pass to RAND for random
					weights seeding
	
	* random connections	-	The /probability/ factor of a connection
					being made between two nodes in the
					network.

					(Note that no support exists at present
					for restricting the connectivity at a
					layer by layer level, although this may
					be done by combining two or more network
					objects and handling activation from one
					to the next programmatically)

	* learning algorithm	-	the learning algorithm (this must be a
					valid compatible perl module).

	* networktype		-	E.g. feedforward. Must be a compatible
					perl module.

	* debug			-	level of debug information
					a different debug level is assigned to
					each module type, and the debug property is
					an ARRAY of which debugs you require.
					0 - error
					1 - TBD
					2 - nnflex core debug
					3 - networktype debug
					4 - learning algorithm debug
					5 - activation function debug
					6 - GUI/Graphic

Plus other custom settings used in networktype & learning algorithm modules, such as:

	* learning rate		-	A constant for use in e.g. backprop learning



Returns a network object that contains $$network->{'layers'}
which is an array of 'layer' objects. 

The layer object contains a property 'nodes' which is an array of nodes
in that layer. So programmatically if you want to access a particular
node (or to interact with the mesh for writing networktypes and learning
algorithms) you can access any node directly using the syntax

$network->{'layers'}->[layer number]->{'nodes'}->[node number]->{property}

(HINT: or do foreach's on the arrays)

Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.
=cut

###############################################################################
sub new
{
	my $class = shift;
	my $network={};
	bless $network,$class;

	my $params = shift;
	my $netParams = shift;
	my @layers;
	dbug ($netParams,"Entered AI::nnflex::new with params $params $netParams",2);

	# Network wide parameters (e.g. random weights)
	foreach (keys %$netParams)
	{
		$network->{$_} = ${$netParams}{$_};
	}

	# one of the parameters will normally be networktype
	# this specifies a class for activation flow, which
	# is included here
	if( $network->{'networktype'})
	{
		my $requirestring = "require \"AI/nnflex/".$network->{'networktype'}.".pm\"";
		eval($requirestring);
	}
	if( $network->{'learning algorithm'})
	{
		my $requirestring = "require \"AI/nnflex/".$network->{'learning algorithm'}.".pm\"";
		eval($requirestring);
	}
	if( $network->{'debug'})
	{
		@DEBUG = @{$network->{'debug'}};
	}

	# build the network
	foreach (@$params)
	{
		if (!($$_{'nodes'})){next}
		my %layer = %{$_};	
		push @layers,AI::nnflex::layer->new(\%layer);	
	}	
	$$network{'layers'} = \@layers;


	# implement the bias node

	if ($network->{'bias'})
	{
		my $biasNode = AI::nnflex::node->new({'activation function'=>'linear'});
		$$network{'biasNode'} = $biasNode;
		$$network{'biasNode'}->{'activation'} = 1;
	}



	$network->init;
	return $network;
}


###############################################################################
# AI::nnflex::output
###############################################################################
=head 

=head1 AI::nnflex::output


$object->output({"output"=>"1"}); returns the activation of layer 1


else returns activation of last layer as a reference to an array

=cut

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

	my $output = AI::nnflex::layer::layer_output($outputLayer);
	return $output;
}
################################################################################
# sub init
################################################################################

=pod

=head1 AI::nnflex::init



called from AI::nnflex::new. no external use required, but not defined as local, in case of debugging use

Init runs through each layer of node objects, creating properties in each node:

	* connectedNodesEast	-	Nodes connected to this node in the layer
					to the 'east', i.e. next layer in feedforward
					terms.

	* connectedNodesWest	-	Nodes connected to this node in the layer to
					the west, i.e. previous layer in feedforward
					networks.

These properties are hashes, with the node object acting as a key. Each value
is a weight for this connection. This means that you /can/ have connection weights
for connections in both directions, since the weight is associated with an incoming
connection.

access with the following syntax:

$node->{'connectedNodesWest'}->{'weights'}->{$connectedNode} = 0.12345
$node->{'connectedNodesWest'}->{'nodes'}->[number] = $nodeObject


Note also that the probability of a connection being created is equal to the numeric 
value of the global property 'random connections' expressed as a decimal between 0 and 1.
If 'random connections' is not specified all connections will be made.

The connections are /only/ created from west to east. Connections that already exist from
west to east are just copied for the 'connectedNodesWest' property.

No return value: the connections are created in the $network object.


These connectedNodesEast & West are handy, because they are arrays you can foreach
them to iterate through the connected nodes in a layer to the east or west.

=cut

###############################################################################
sub init
{

	#Revised version of init for nnflex

	my $network = shift;
	my @layers = @{$network->{'layers'}};

	my $nodeid = 1;
	my $currentLayer;	
	# foreach layer, we need to examine each node
	foreach my $layer (@layers)
	{
		# Foreach node we need to make connections east and west
		foreach my $node (@{$layer->{'nodes'}})
		{
			# import the nodes activation function
			my $requireString = "require (\"AI/nnflex/".$node->{'activation function'}.".pm\")";
			eval ($requireString);
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
							if ($network->{'random weights'})
							{
								$weight = rand(2)-1;
							}
							else
							{
								$weight = 0;
							}
							push @{$node->{'connectedNodesWest'}->{'nodes'}},$westNodes;
							${$node->{'connectedNodesWest'}->{'weights'}}{$westNodes} = $weight;
							#${$node->{'connectedNodesWest'}}{$connectionFromWest} = $weight;
							#my $westNodeRef = \$westNodes;
							#${$node->{'connectedNodesWest'}}{$westNodeRef} = $weight;
							
							#if ($network->{'bias'} && $currentLayer ==0)
							#if ($network->{'bias'} && !$alreadyBiased)
							#{
							#	push @{$node->{'connectedNodesWest'}->{'nodes'}},$network->{'biasNode'};
							#	$network->{'biasNode'}->{'activation'}=1;
							#	$network->dbug ("West to east Connection - ".$network->{'biasNode'}." to $node",2);
							#	$alreadyBiased=1; # make sure only 1 bias connection per node
							#}
								
							$network->dbug ("West to east Connection - $westNodes to $node",2);
						}
					}
				}
			}

			# Now initialise connections to the east (if not last layer)
			if ($currentLayer < (scalar @layers)-1)
			{
			foreach my $eastNodes (@{$network->{'layers'}->[$currentLayer+1]->{'nodes'}})
			{
				if (!$network->{'random connections'}  || $network->{'random connections'} > rand(1))
				{
					my $weight;
					if ($network->{'random weights'})
					{
						$weight = rand(1);
					}
					else
					{
						$weight = 0;
					}
					push @{$node->{'connectedNodesEast'}->{'nodes'}},$eastNodes;
					${$node->{'connectedNodesEast'}->{'weights'}}{$eastNodes} = $weight;
					$network->dbug ("East to west Connection $node to $eastNodes",2);
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
				if ($network->{'random weights'})
				{
					$weight = rand(1);
				}
				else
				{
					$weight = 0;
				}
				${$node->{'connectedNodesWest'}->{'weights'}}{$network->{'biasNode'}} = $weight;

				$network->dbug ("West to east Connection - ".$network->{'biasNode'}." to $node weight = $weight",2);
			}
		}
	}






}

###############################################################################
# sub $network->dbug
###############################################################################

=pod

=head1 AI::nnflex::$network->dbug



Internal use, writes to STDOUT parameter 1 if parameter 2 == global variable $DEBUG.
or parameter 2 == 1

=cut

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
# AI::nnflex::dump_state
###############################################################################

=pod

=head1 AI::nnflex::dump_state



$network->dump_state({"filename"=>"test.wts"[, "activations"=>1]});

Dumps the current contents of the node weights to a file.

=cut

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
=pod

=head1 load_state

useage:
	$network->load_state(<filename>);

Initialises the network with the state information (weights and, optionally
activation) from the specified filename.

Note that if you have a file containing activation, but you are not using
persistent activation, the activation states of nodes will be reset during
network->run

=cut
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





###############################################################################
###############################################################################
# Package AI::nnflex::layer
###############################################################################
###############################################################################
package AI::nnflex::layer;

=pod

=head1 AI::nnflex::layer



The layer object

=cut

###############################################################################
# AI::nnflex::layer::new
###############################################################################

=pod

=head1 AI::nnflex::layer::new



Create new layer

Takes the parameters from AI::nnflex::layer and passes them through to AI::nnflex::node::new

(Uses nodes=>X to decide how many nodes to create)

Returns a layer object containing an array of node objects

=cut

###############################################################################
sub new
{
	my $layer={};
	my $class = shift;
	my $params = shift;
	 
	my $numNodes = $$params{'nodes'};
	
	my @nodes;

	for (1..$numNodes)
	{
		push @nodes, AI::nnflex::node->new($params);
	}

	$$layer{'nodes'} = \@nodes;
	bless $layer,$class;
	AI::nnflex::dbug($params,"Created layer $layer",2);
	return $layer;
}

###############################################################################
# AI::nnflex::layer::layer_output
##############################################################################

=pod

=head1 AI::nnflex::layer::layer_output



Receives a reference to a hash of parameters. Valid inputs are

	* layer		-	the layer number you want output from

Returns a reference to an array of outputs

Used by AI::nnflex::output

=cut

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


###############################################################################
###############################################################################
# package AI::nnflex::node
###############################################################################
###############################################################################
package AI::nnflex::node;

=pod

=head1 AI::nnflex::node



the node object

=cut

###############################################################################
# AI::nnflex::node::new
###############################################################################

=pod

=head1 AI::nnflex::node::new



Takes parameters passed from nnflex via AI::nnflex::layer

returns a node object containing:

	* activation		-	the nodes current activation

	* decay			-	decay rate

	* adjust error		-	NYI

	* persistent activation -	if true, activation will be summed
					on each run, rather than zeroed
					before calculating inputs.

	* ID			-	node identifier (unique across the
					nnflex object)

	* threshold		-	the level above which the node
					becomes active

	* activation function	-	the perl script used as an activation
					function. Must perform the calculation
					on a variable called $value.

	* active		-	whether the node is active or
					not. For lesioning. Set to 1
					on creation



=cut

###############################################################################
sub new
{
	my $class = shift;
	my $node = {};
	my $params = shift;

	if ($$params{'random activation'})
	{
		$$node{'activation'} = 
			rand($$params{'random'});
			AI::nnflex::dbug($params,"Randomly activated at ".$$node{'activation'},2);
	}
	else
	{
		$$node{'activation'} = 0;
	}
	$$node{'random weights'} = $$params{'random weights'};
	$$node{'decay'} = $$params{'decay'};
	$$node{'adjust error'} = $$params{'adjust error'};
	$$node{'persistent activation'} = $$params{'persistent activation'};
	$$node{'threshold'} = $$params{'threshold'};
	$$node{'activation function'} = $$params{'activation function'};
	$$node{'active'} = 1;
	

	# we dont do this anymore
	#load the activation function code into an attribute
	#open (ACT,"nnflex/".$$node{'activation function'}.".pl") or die "Can't open activation function file\n";
	#while (<ACT>)
	#{
	#	$$node{'activation function code'} .= $_;
	#}
	#close ACT;


	$$node{'error'} = 0;
	
	bless $node,$class;	
	AI::nnflex::dbug($params,"Created node $node",2);
	return $node;
}


1;

