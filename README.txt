#######################################################
#######################################################
NNFlex - a flexible native perl API for neural networks

Author: Charles Colbourn
Version: 0.12
Date: January 2005

NNFLEX is intended to be a flexible framework for
developing neural network applications. The code is
moderately complex and not really suitable for learning
the basic functioning of the backpropagation algorithm.
Use Phil Brierleys excellent example code at
http://www.philbrierley.com
for that (it's in java and vb on the site, If you need
to read it in perl, there is a translated version in
this package, filename bp.pl).

This document gives much more detail about the API
than the standard POD (although if you dig around in the
POD its all in there somewhere). I habitually document
every method, public or private, with a block of POD 
beforehand, so the code should be very easy to understand.


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

The current version of NNFlex:: is basically just an API,
rather than a modelling package. A more user friendly
frontend may follow later. To create a network, use the
following syntax:

use AI::NNFlex;
my $network = AI::NNFlex->new([array of hashes],{hash of global config});

for example:

my $object = AI::NNFlex->new([{"nodes"=>2,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation fu
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

The object returned is an NNFlex object, which contains
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

The NNFlex object has a property called 'layers'. This
is an array containing an AI::NNFlex::layer object for each
layer in the network.

The AI::NNFlex::layer object has a property called 'nodes'.
This is an array containing AI::NNFlex::node objects, one
for each node in the layer.

The AI::NNFlex::node object has a whole host of properties,
see the perldoc for full details. Among them are:
* activation 	- the current activation of the node
* activation function
	 	- the perl module used to calculate
		  the nodes activation.
* id		- the numeric ID of the node
* error		- the current network error of the node
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

ID is a purely numeric incremental ID for the node. Please
don't do anything programmatic with this, its just for
human reference.

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
0.11

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
Please feel free to amend, improve and patch NNFlex:: to 
whatever extent you wish. Any really nice additional
functionality, please mail it to me (as a diff, an 
amended module/script, or a description) at the address
below.


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
Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify it under the same terms as Perl itself.


CONTACT

charlesc@nnflex.g0n.net



