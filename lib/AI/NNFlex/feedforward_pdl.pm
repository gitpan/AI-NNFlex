##########################################################
# AI::NNFlex::feedforward_pdl
##########################################################
# This is a new version of feedforward using PDL for
# speed
#
##########################################################
# Versions
# ========
#
# 1.0	20050219	CColbourn	New module
#
##########################################################
# ToDo
# ----
#
###########################################################
#
use strict;
use PDL;

##########################################################
# AI::NNFlex::feedforward
##########################################################
=pod
=head1 AI::NNFlex::feedforward_pdl - feedforward module for NNFlex

=item

This module is the feedforward network type for NNFlex. it
is included in the NNFlex:: namespace at run time. See documentation
below for standard methods.

Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=cut





###########################################################
# AI::NNFlex::feedforward::run
###########################################################
=pod

=head1 AI::NNFlex::feedforward::run

=item

This class contains the run method only. The run method performs
feedforward  (i.e. west to east) activation flow on the network.

This class is internal to the NNFlex package, and is included
in the NNFlex namespace by a require on the networktype parameter.

syntax:
 $network->run([0,1,1,1,0,1,1]);

=cut
###########################################################
sub run
{
	my $network = shift;

	my $inputPatternRef = shift;
	
	# if this is an incorrect dataset call translate it
	if ($inputPatternRef =~/Dataset/)
	{
		return ($inputPatternRef->run($network))
	}


	my @inputPattern = @{$inputPatternRef};

	# If the network is biased, push a 1 onto the input pattern
	if ($network->{'bias'})
	{
		push @inputPattern,1;
	}


	my $inputPattern = pdl(@inputPattern);

	my @debug = @{$network->{'debug'}};
	if (scalar @debug> 0)
	{$network->dbug ("Input pattern $inputPattern received by feedforward",3);}


	# First of all apply the activation pattern to the input layer
	# and create the layer piddle

	my $inputLayer = $network->{'layers'}->[0];

	# Put this back in when we've figured out how to work out
	# the width of a piddle
	#if (scalar @$inputLayer != scalar @inputPattern)
	#{	
	#	$network->dbug("Wrong number of input values",0);
	#	return 0;
	#}

	$inputLayer->{'pdlActivation'} = $inputPattern;

	# To make sure this works with non pdl backprop (temporary)
	# we have to set the activations on the input layer nodes
	my $counter=0;
	foreach my $node (@{$inputLayer->{'nodes'}})
	{
		$node->{'activation'} = $inputPattern[$counter++];
	}


	# Now flow activation through the network starting with the second layer
	my $counter=0;
	foreach my $layer (@{$network->{'layers'}})
	{
		if ($layer eq $network->{'layers'}->[0]){ $counter++;next}
		my @layerActivations;

		my $inputs = $network->{'layers'}->[$counter-1]->{'pdlActivation'};

		foreach my $node (@{$layer->{'nodes'}})
		{
			my $weights = pdl($node->{'connectedNodesWest'}->{'weights'});


			my $adjustedInputs = $weights * $inputs;

			my $activation = sum($adjustedInputs);
			my $function = $node->{'activationfunction'};
			my $functionCall ="\$activation = \$network->$function(\$activation);";
			eval($functionCall);


			# if the node is inactive blank the activation
			if (!$node->{'active'})
			{
				$activation = 0;
			}			


			$node->{'activation'} = $activation;

			push @layerActivations,$activation;
			if (scalar @debug> 0)
			{$network->dbug("Final activation of $node = ".$node->{'activation'},3);}
		}
		if ($network->{'bias'})
		{
			push @layerActivations,1;
		}


		$layer->{'pdlActivation'} = pdl(@layerActivations);

		$counter++;

	}






}





1;

