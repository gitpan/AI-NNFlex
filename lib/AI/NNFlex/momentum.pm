##########################################################
# AI::NNFlex::momentum
##########################################################
# Backprop with simple (non adaptive) momentum
##########################################################
# Versions
# ========
#
# 1.0	20050121	CColbourn	New module
# 1.1	20050201	CColbourn	Added call to activation
#					function slope instead
#					of hardcoded 1-y*y
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
# AI::NNFlex::momentum::learn
###########################################################
=pod

=head1 AI::NNFlex::momentum

=item

This module is the momentuma algorithm for NNFlex. it
is included in the NNFlex namespace at run time. See documentation
below for standard methods.

The momentum module is a modified version of the backprop
module (literally - I copied it then did '1,$s/backprop/momentum/g'!)

The only difference is that momentum retains a copy of node dW 
at the end of the learning cycle. This is then used in the dW
calculation for the next pass. The upshot of this is that if a
large change took place last time, we're evidently still in the
'large changes' stage of learning, so it should be a large change
this time as well.

Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=cut

###########################################################
#AI::NNFlex::momentum::learn
###########################################################
=pod

=head1 AI::NNFlex::momentum::learn

=item

Takes as a parameter a reference to the desired output
pattern, performs one learning pass back through the network 
with normal momentum procedures to bring the network closer 
to convergence.

This package is imported into the NNFlex namespace at runtime
via a parameter to the network object.

syntax:
 $network->learn([0,1,1,0]);

=cut

###########################################################
# AI::NNFlex::momentum::calc_error
###########################################################
=pod
=cut

sub calc_error
{
	my $network = shift;

	my $outputPatternRef = shift;
	my @outputPattern = @$outputPatternRef;

	my @debug = @{$network->{'debug'}};

	if (scalar @debug > 0)
	{$network->dbug ("Output pattern @outputPattern received by momentum",4);}


	my $outputLayer = $network->{'layers'}->[-1]->{'nodes'};

	if (scalar @$outputLayer != scalar @outputPattern)
	{	
		$network->dbug ("Wrong number of output values, net has ".scalar @$outputLayer." nodes",0);
		return 0;
	}

	# Now calculate the error
	my $counter=0;
	foreach (@$outputLayer)
	{	
		my $value = $_->{'activation'} - $outputPattern[$counter];
		$_->{'error'} = $value;
		$counter++;
		if (scalar @debug > 0)
		{$network->dbug ("Error on output node $_ = ".$_->{'error'},4);}
	}


}


########################################################
# AI::NNFlex::momentum::learn
########################################################
=pod

=head1 AI::NNFlex::momentum::learn

=item

learn is the main method of the momentum module. It calls
calc_error to calculate the output error, calls output_adjust
to adjust the weights from the last hidden to the output
layer, then calls hidden_adjust to adjust the weights to the
hidden layers
finally it returns the network sqrd error.

=cut
#########################################################
sub learn
{

	my $network = shift;

	my $outputPatternRef = shift;

	# if this is an incorrect dataset call translate it
	if ($outputPatternRef =~/Dataset/)
	{
		return ($outputPatternRef->learn($network))
	}


	my @outputPattern = @$outputPatternRef;

	$network->calc_error($outputPatternRef);

	#calculate & apply dWs
	$network->hiddenToOutput;
	if (scalar @{$network->{'layers'}} > 2) 
	{
		$network->hiddenOrInputToHidden;
	}

	# calculate network sqErr
	my $Err = $network->RMSErr($outputPatternRef);
	return $Err;	
}


#########################################################
# AI::NNFlex::momentum::hiddenToOutput
#########################################################
=pod

=head1 AI::NNFlex::momentum::hiddenToOutput

=item
Performs weight changes for all nodes in the output layer
nodes 'connectedNodesWest' attributes, based on momentum
Output weights delta

=cut
##########################################################
sub hiddenToOutput
{
	my $network = shift;

	my @debug = @{$network->{'debug'}};

	my $outputLayer = $network->{'layers'}->[-1]->{'nodes'};

	foreach my $node (@$outputLayer)
	{
		foreach my $connectedNode (@{$node->{'connectedNodesWest'}->{'nodes'}})
		{
			my $momentum = 0;
			if ($node->{'connectedNodesWest'}->{'lastdelta'}->{$connectedNode})
			{
				$momentum = ($network->{'momentum'})*($node->{'connectedNodesWest'}->{'lastdelta'}->{$connectedNode});
			}
			if (scalar @debug > 0)
			{$network->dbug("Learning rate is ".$network->{'learningrate'},4);}
			my $deltaW = (($network->{'learningrate'}) * ($node->{'error'}) * ($connectedNode->{'activation'}));
			$deltaW = $deltaW+$momentum;
			$node->{'connectedNodesWest'}->{'lastdelta'}->{$connectedNode} = $deltaW;
			
			if (scalar @debug > 0)
			{$network->dbug("Applying delta $deltaW on hiddenToOutput $connectedNode to $node",4);}
			# 
			$node->{'connectedNodesWest'}->{'weights'}->{$connectedNode} -= $deltaW;
		}
			
	}
}

######################################################
# AI::NNFlex::momentum::hiddenOrInputToHidden
######################################################
=pod

=head1 AI::NNFlex::momentum::hiddenOrInputToHidden

=item
This subroutine calculates and applies delta weights that are not directly
derived from the output layer. I.e. if you have a 2 layer network this
will never get called.

If you have a 3 layer network this will be called to delta the weights
between input and hidden layers.

If you have a 4 layer network it will be called to delta the weights between
the two hidden layers, and the input to hidden layer

=cut

#######################################################
sub hiddenOrInputToHidden
{

	my $network = shift;

	my @layers = @{$network->{'layers'}};

	my @debug = @{$network->{'debug'}};

	# remove the last element (The output layer) from the stack
	# because we've already calculated dW on that
	pop @layers;

	if (scalar @debug > 0)
	{$network->dbug("Starting momentum of error on ".scalar @layers." hidden layers",4);}

	foreach my $layer (reverse @layers)
	{
		foreach my $node (@{$layer->{'nodes'}})
		{
			if (!$node->{'connectedNodesWest'}) {last}

			my $nodeError;
			foreach my $connectedNode (@{$node->{'connectedNodesEast'}->{'nodes'}})
			{
				$nodeError += ($connectedNode->{'error'}) * ($connectedNode->{'connectedNodesWest'}->{'weights'}->{$node});
			}
			if (scalar @debug > 0)
			{$network->dbug("Hidden node $node error = $nodeError",4);}
			$node->{'error'} = $nodeError;


			# update the weights from nodes inputting to here
			foreach my $westNodes (@{$node->{'connectedNodesWest'}->{'nodes'}})
			{
				
				my $momentum = 0;
				if($node->{'connectedNodesWest'}->{'lastdelta'}->{$westNodes})
				{
					$momentum = ($network->{'momentum'})*($node->{'connectedNodesWest'}->{'lastdelta'}->{$westNodes});
				}

				# get the slope from the activation function component
				my $value = $node->{'activation'};
				my $evalstring = "\$value = \$network->".$node->{'activationfunction'}."_slope(\$value);";
				eval($evalstring);


				$value = $value * $node->{'error'} * $network->{'learningrate'} * $westNodes->{'activation'};

				
				my $dW = $value;
				$dW = $dW + $momentum;
				if (scalar @debug > 0)
				{$network->dbug("Applying deltaW $dW to inputToHidden connection from $westNodes to $node",4);}

				$node->{'connectedNodesWest'}->{'lastdelta'}->{$westNodes} = $dW;

				$node->{'connectedNodesWest'}->{'weights'}->{$westNodes} -= $dW;
				if (scalar @debug > 0)
				{$network->dbug("Weight now ".$node->{'connectedNodesWest'}->{'weights'}->{$westNodes},4);}
			}	


		}
	}
								
				

}

#########################################################
# AI::NNFlex::momentum::sqrErr
#########################################################
=pod

=head1 AI::NNFlex::momentum::sqrErr

=item

Calculates the network squared error after a single
momentum pass. Internal to ::momentum

=cut
##########################################################
sub RMSErr
{
	my $network = shift;

	my $outputPatternRef = shift;
	my @outputPattern = @$outputPatternRef;

	my @debug = @{$network->{'debug'}};

	my $sqrErr;

	my $outputLayer = $network->{'layers'}->[-1]->{'nodes'};

	if (scalar @$outputLayer != scalar @outputPattern)
	{	
		$network->dbug("Wrong number of output values, net has ".scalar @$outputLayer." nodes",0);
		return 0;
	}

	# Now calculate the error
	my $counter=0;
	foreach (@$outputLayer)
	{	
		my $value = $_->{'activation'} - $outputPattern[$counter];

		$sqrErr += $value *$value;
		$counter++;
		if (scalar @debug > 0)
		{$network->dbug("Error on output node $_ = ".$_->{'error'},4);}
	}

	my $error = sqrt($sqrErr);

	return $error;
}

1;

