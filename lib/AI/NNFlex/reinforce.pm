##########################################################
# AI::NNFlex::reinforce
##########################################################
# NNFlex learning module
# this is a bit of an experimental one. All it does is
# reinforce the weight depending on the sign & activity
# of the node, sort of a gross oversimplification of a
# neuron.
#
##########################################################
# Versions
# ========
#
# 1.0	20041125	CColbourn	New module
# 1.1	20050116	CColbourn	Fixed reverse @layers
#					bug reported by GM Passos
#
##########################################################
# ToDo
# ----
#
#
###########################################################
#
use strict;

###########################################################
# AI::NNFlex::reinforce
###########################################################
=pod

=head1 AI::NNFlex::reinforce

=item

This module is a (very experimental) reinforcement learning
algorithm for NNFlex. It works on the basis of a GROSS simplification
of neuron activity, in that it reinforces active paths. It is
is included in the NNFlex namespace at run time. See documentation
below for standard methods.

Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=cut



###########################################################
#AI::NNFlex::reinforce::learn
###########################################################
=pod

=head1 AI::NNFlex::reinforce::learn

=item

Reinforces weights by activity * learning rate.

Performs one learning pass back through the network - only
currently reinforces from west to east (i.e. no reverse
flow).

This package is imported into the NNFlex namespace at runtime
via a parameter to the network object.

syntax:
 $network->learn();

=cut

#########################################################
sub learn
{

	my $network = shift;

	my @layers = @{$network->{'layers'}};

	# no connections westwards from input, so no weights to adjust
	shift @layers;

	# reverse to start with the last layer first
	foreach my $layer (reverse @layers)
	{
		my @nodes = @{$layer->{'nodes'}};

		foreach my $node (@nodes)
		{
			my @westNodes = @{$node->{'connectedNodesWest'}->{'nodes'}};
			my %westWeights = %{$node->{'connectedNodesWest'}->{'weights'}};
			foreach my $westNode (@westNodes)
			{
				my $dW = $westNode->{'activation'} * $westWeights{$westNode} * $network->{'learning rate'};
				$node->{'connectedNodesWest'}->{'weights'}->{$westNode} += $dW;
			}
		}
	}
}



1;

