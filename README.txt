#######################################################
#######################################################
NNFlex - a flexible native perl API for neural networks

Author: Charles Colbourn
Version: 0.15
Date: February 2005

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

use AI::NNFlex;
my $network = AI::NNFlex->new(config item=>value);

for example:
my $network = AI::NNFlex->new(randomconnections=>0,
				networktype=>'feedforward',
				randomweights=>1,
				learningalgorithm=>'momentum',
				learningrate=>.1,
				debug=>[3,4],bias=>1,
				momentum=>0.6);

Then add layers:

$network->add_layer(    nodes=>2,
			persistentactivation=>0,
			decay=>0.0,
			randomactivation=>0,
			threshold=>0.0,
			activationfunction=>"tanh",
			randomweights=>1);

Initialize the network to create connections between nodes:

$network->init();

And you have a usable network. You then need to train the network. 
There are two ways of doing this, the batch method, and the single
run and learn. To train the network on a single pattern, do this:

$network->run([0,1,0,1]);
$network->learn([0,1]);

To get back the output patterns without performing a learning
pass, use:

my $resultArrayRef = $network->run([0,1,0,1]);

(Note that the run and learn methods are dependent upon the 
network type you have imported in the constructor. The example
above is a feedforward network using backprop with momentum. 
Other importable modules are planned, including Hopfield, 
Boltzmann & Gamma networks, and Quickprop learning for FF
nets).

Batch learning involves creating a dataset:

my $dataset = AI::NNFlex::Dataset->new([0,1,0,1],[1,1],
					[1,1,0,0],[0,1],
					[1,0,0,1],[1,0]);

and applying the dataset to the network:

my $networkError = $dataset->learn($network);

You can get an array of arrays of outputs without the
learning operation using:

my $arrayRef = $dataset->run($network);

which come back in the same order as the dataset is created
in. Or you can run a single pattern through as per the single
method.

If you want to experimentally lesion the network you can
call the $network->lesion(probability) method, there are
corresponding methods for layers and single nodes.

NB All probabilities in NNFlex are expressed as a value
between 0 and 1.



Rationale:

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
0.16

ACKNOWLEDGEMENTS
Phil Brierley, for making his excellent example code
(which dug me out of a backprop hole) available to 
all and sundry.

Josiah Bryan, for writing the original native perl
backprop modules, which I've used for inspiration in
the way the API interface is constructed.

Dr Martin Le Voi, for considerable amounts of help when
trying to get my head around the fundamentals of backprop.

Dr David Plaut, who gave me unstinting and timely help 
getting Xerion running when I realised this code wasn't
going to be finished in time for my MSc project.


BUGS, FEATURE REQUESTS AND PATCHES
Please feel free to amend, improve and patch NNFlex to 
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
improved speed!
Tk based GUI frontend
gamma network module (DHYB)
Hopfield module
quickprop learning
Boltzmann module
PDL option for faster execution


COPYRIGHT
Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify it under the same terms as Perl itself.


CONTACT

charlesc@nnflex.g0n.net



