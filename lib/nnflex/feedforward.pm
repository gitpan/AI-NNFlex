##########################################################
# AI::nnflex::feedforward
##########################################################
# This is the first propagation module for nnflex
#
##########################################################
# Versions
# ========
#
# 1.0	20040910	CColbourn	New module
#
##########################################################
# ToDo
# ----
#
# This needs to implement at least a 'run' method that 
#  propagates activation through the network from west to 
#  east.
#
###########################################################
#
use strict;

##########################################################
# AI::nnflex::feedforward
##########################################################
=pod
=head1 AI::nnflex::feedforward - feedforward module for nnflex

=item

This module is the feedforward network type for nnflex. it
is included in the nnflex namespace at run time. See documentation
below for standard methods.

Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=cut





###########################################################
# AI::nnflex::feedforward::run
###########################################################
=pod

=head1 AI::nnflex::feedforward::run

=item

This class contains the run method only. The run method performs
feedforward  (i.e. west to east) activation flow on the network.

This class is internal to the nnflex package, and is included
in the nnflex namespace by a require on the networktype parameter.

syntax:
 $network->run([0,1,1,1,0,1,1]);

=cut
###########################################################
sub run
{
	my $network = shift;

	my $inputPatternRef = shift;
	my @inputPattern = @$inputPatternRef;

	$network->dbug ("Input pattern @inputPattern received by feedforward",3);


	# First of all apply the activation pattern to the input units (checking
	# that the pattern has the right number of values)

	my $inputLayer = $network->{'layers'}->[0]->{'nodes'};

	if (scalar @$inputLayer != scalar @inputPattern)
	{	
		$network->dbug("Wrong number of input values",0);
		return 0;
	}

	# Now apply the activation
	my $counter=0;
	foreach (@$inputLayer)
	{	
		if ($_->{'active'})
		{

			if ($_->{'persistent activation'})
			{
				$_->{'activation'} +=$inputPattern[$counter];
				$network->dbug("Applying ".$inputPattern[$counter]." to $_",3);
			}
			else
			{
				$_->{'activation'} =$inputPattern[$counter];
				$network->dbug("Applying ".$inputPattern[$counter]." to $_",3);
			 
			}
		}
		$counter++;
	}
	

	# Now flow activation through the network starting with the second layer
	foreach my $layer (@{$network->{'layers'}})
	{
		if ($layer eq $network->{'layers'}->[0]){next}

		foreach my $node (@{$layer->{'nodes'}})
		{

			if (!($node->{'persistent activation'}))
			{
				$node->{'activation'} =0;
			}

			foreach my $connectedNode (@{$node->{'connectedNodesWest'}->{'nodes'}})
			{
				$network->dbug("Flowing from $connectedNode to $node",3);
	
				my $weight = ${$node->{'connectedNodesWest'}->{'weights'}}{$connectedNode};
				my $activation = $connectedNode->{'activation'};		
				$network->dbug("Weight & activation: $weight - $activation",3);
				
				# Decay the node (note that if decay is not set this
				# will have no effect, hence no if).
				$node->{'activation'} -= $node->{'decay'};

				my $totalActivation = $weight*$activation;
				#my $value = $totalActivation;
				my $function = $node->{'activation function'};
				my $functionCall ="\$totalActivation = \$network->$function(\$totalActivation);";
				eval($functionCall);
				$node->{'activation'} +=$totalActivation; 
	
			}

			if ($node->{'active'})
			{
				my $value = $node->{'activation'};
				#eval($node->{'activation function code'});
				my $function = $node->{'activation function'};
				my $functionCall ="\$value = \$network->$function(\$value);";

				eval($functionCall);
				#my $value = $node->'$function'($network,$value);
				$node->{'activation'} = $value;
			}
			$network->dbug("Final activation of $node = ".$node->{'activation'},3);
		}
	}





}





1;

